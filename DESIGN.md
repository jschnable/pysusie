# pysusie: Design Specification

## 1. Overview

**pysusie** is a Python package implementing the Sum of Single Effects (SuSiE)
model for Bayesian variable selection in linear regression. It is based on the
SuSiE methodology described in three key publications:

- **Wang et al. (2020)** — The original SuSiE model and IBSS algorithm
- **Zou et al. (2022)** — SuSiE-RSS: extension to summary statistics
- **McCreight et al. (2025)** — SuSiE 2.0: modular redesign with SuSiE-ash/inf

The package targets genetics/genomics fine-mapping as its primary use case but
is general-purpose for any sparse Bayesian variable selection problem.

---

## 2. Mathematical Foundation

### 2.1 Core Model (SuSiE)

The SuSiE model decomposes the effect vector as a sum of L "single effects":

```
y = X b + e,       e ~ N(0, sigma^2 I_n)
b = sum_{l=1}^{L} b_l
b_l = gamma_l * beta_l      (single effect vector)
gamma_l ~ Multinomial(1, pi)  (one-hot: exactly one variable has the effect)
beta_l ~ N(0, sigma_{0l}^2)  (effect size prior)
```

Where:
- `y`: n-vector of responses
- `X`: n x p matrix of predictors
- `b`: p-vector of regression coefficients
- `L`: number of single effects (user-specified upper bound)
- `gamma_l`: indicator vector (which variable carries effect l)
- `pi`: p-vector of prior inclusion probabilities
- `sigma_{0l}^2`: prior variance for effect l

### 2.2 Variational Inference

The posterior is approximated by a fully factorized distribution:

```
q(b_1, ..., b_L) = prod_{l=1}^{L} q_l(b_l)
```

Each `q_l` is a mixture of point mass at zero and a Gaussian:

```
q_l(b_l) = sum_j alpha_{lj} * N(mu_{lj}, sigma_{lj}^2) * delta(b_{l,-j} = 0)
```

Key posterior quantities per effect l and variable j:
- `alpha[l,j]`: posterior inclusion probability (PIP for effect l at variable j)
- `mu[l,j]`: posterior mean of effect size (conditional on inclusion)
- `mu2[l,j]`: posterior second moment (conditional on inclusion)

Marginal PIP for variable j (combining all effects):
```
PIP_j = 1 - prod_{l=1}^{L} (1 - alpha[l,j])
```

### 2.3 IBSS Algorithm (Iterative Bayesian Stepwise Selection)

The core fitting algorithm is coordinate ascent on the ELBO:

```
For iteration = 1 to max_iter:
    For l = 1 to L:
        1. Compute expected residuals: r_l = y - X * sum_{l' != l} E[b_{l'}]
        2. Fit Single Effect Regression (SER) on r_l:
           For each variable j:
             betahat_j = (x_j' r_l) / (x_j' x_j)
             shat2_j   = sigma^2 / (x_j' x_j)
             lbf_j     = -0.5 * log(1 + V_l/shat2_j)
                        + 0.5 * betahat_j^2 * V_l / (shat2_j * (shat2_j + V_l))
           alpha[l,:] = softmax(lbf + log(pi))
           mu[l,j]    = V_l / (V_l + shat2_j) * betahat_j
           mu2[l,j]   = V_l * shat2_j / (V_l + shat2_j) + mu[l,j]^2
        3. Optionally estimate V_l (prior variance) by empirical Bayes
        4. Update fitted values
    Estimate sigma^2 if requested
    Check convergence (ELBO change < tol)
```

### 2.4 Evidence Lower Bound (ELBO)

```
ELBO = E_q[log p(y | X, b, sigma^2)] - sum_l KL(q_l || p_l)

E_q[log p(y|...)] = -n/2 * log(2*pi*sigma^2) - 1/(2*sigma^2) * E[||y - Xb||^2]

E[||y - Xb||^2] = y'y - 2 * sum_j (sum_l alpha[l,j]*mu[l,j]) * (X'y)_j
                 + sum_j d_j * sum_l alpha[l,j] * mu2[l,j]
                 + cross terms between effects
```

where `d_j = x_j' x_j` (diagonal of X'X).

### 2.5 Summary Statistics Variant (SuSiE-RSS)

When only z-scores and an LD (correlation) matrix R are available:

```
Sufficient statistics mapping (standardized data):
  X'X ≈ (n-1) * R
  X'y ≈ sqrt(n-1) * z_adjusted
  y'y ≈ n-1

PVE-adjusted z-scores (when n is known):
  z_adj_j = sqrt((n-1) / (z_j^2 + n - 2)) * z_j

LD regularization (when R is not from in-sample data):
  R_tilde = (1 - lambda) * R + lambda * I
  lambda estimated by maximizing N(z_adj; 0, R_tilde)
```

### 2.6 SuSiE-ash Extension (Adaptive Shrinkage Background)

Models both sparse causal effects and polygenic background:

```
y = X * beta_sparse + X * theta_background + epsilon

beta_sparse: modeled by standard SuSiE (L single effects)
theta_background: modeled by Mr.ASH (mixture of normals prior)
  theta_j ~ sum_k pi_k * N(0, sigma^2 * sigma_k^2)

Precision matrix: Omega = (sigma^2 I + tau^2 X X')^{-1}
  where tau^2 = sigma^2 * sum_k pi_k * sigma_k^2
```

### 2.7 Mr.ASH (Multiple Regression with Adaptive Shrinkage)

Coordinate ascent variational EM for mixture-of-normals regression:

```
For each variable j (in random order):
  1. Compute: b_j * w_j = r' x_j + beta_j * w_j
  2. For each mixture component k: compute posterior mean mu_jk
  3. Softmax over components: phi_jk (responsibilities)
  4. Update: beta_j = sum_k phi_jk * mu_jk
  5. Update residual: r -= x_j * (beta_j_new - beta_j_old)
  6. Accumulate mixture weight updates
Update sigma^2 and pi
```

---

## 3. Package Architecture

### 3.1 Module Layout

```
pysusie/
    __init__.py              # Public API exports
    susie.py                 # SuSiE class (main entry point)
    _ser.py                  # Single Effect Regression + prior variance estimation
    _ibss.py                 # IBSS algorithm (core loop) + residual variance estimation
    _elbo.py                 # ELBO computation
    _credible_sets.py        # Credible set extraction and purity
    _preprocessing.py        # Data centering, scaling, validation, SS/RSS conversion
    _mrash.py                # Mr.ASH implementation
    _unmappable.py           # SuSiE-inf and SuSiE-ash wrappers
    _numba_kernels.py        # Numba-accelerated hot loops
    _plotting.py             # Visualization (matplotlib)
    _utils.py                # Numerical utilities (logsumexp, etc.)
    _types.py                # Type definitions and dataclasses
    datasets.py              # Example datasets
```

### 3.2 Class Hierarchy

```
SuSiE                       # Main class (estimator)
  |- fit(X, y) -> self      # Individual-level data (returns self for sklearn)
  |- fit_from_summary_stats(...) -> self
  |- fit_from_sufficient_stats(...) -> self
  |- predict(X_new)
  |
  |- result_                # SuSiEResult (stored after fitting)
  |- coef_                  # Forwarding property -> result_.coef
  |- intercept_             # Forwarding property -> result_.intercept
  |- pip_                   # Forwarding property -> result_.pip

SuSiEResult                 # Immutable result container (stored in result_ after fit)
  |- alpha, mu, mu2         # Core posterior matrices
  |- pip                    # Marginal PIPs (cached_property)
  |- coef, intercept        # Regression coefficients
  |- prior_variance         # Estimated V per effect
  |- residual_variance      # Estimated sigma^2
  |- elbo                   # ELBO trace
  |- n_iter, converged      # Convergence info
  |- get_credible_sets()    # Compute CS with purity filtering
  |- posterior_mean()       # E[b] = sum_l alpha[l,:] * mu[l,:]
  |- posterior_sd()         # SD of posterior
  |- posterior_samples(n)   # Draw from approximate posterior
  |- lsfr()                 # Local false sign rate
  |- summary()              # Summary DataFrame
  |- plot()                 # PIP / CS visualization

CredibleSet                 # Single credible set
  |- variables: ndarray     # Indices of variables in set
  |- coverage: float        # Achieved coverage
  |- purity: PurityMetrics  # min/mean/median absolute correlation
  |- log_bayes_factor: float
```

### 3.3 Design Principles

1. **scikit-learn compatibility**: `fit()` returns `self` (not a result
   object) so that SuSiE works with Pipeline, GridSearchCV, and other
   sklearn tooling. `predict()` / `coef_` / `intercept_` follow standard
   conventions. Fitted attributes end with underscore.

2. **NumPy-native**: All inputs/outputs are ndarrays. Accept pandas
   DataFrames/Series transparently (store column names for labeling).

3. **Rich result object**: `fit()` stores a `SuSiEResult` in `self.result_`.
   Users access the full result via `model.result_` after fitting.

4. **Estimator-result boundary**: `SuSiEResult` is the canonical data
   container and source of truth. The `SuSiE` estimator exposes thin
   forwarding properties (`coef_`, `intercept_`, `pip_`) that delegate to
   `self.result_.<attr>` for sklearn compatibility. Users who want the full
   result should access `model.result_`.

5. **No R baggage**: Use Pythonic naming (snake_case), standard types,
   and idiomatic patterns. No S3/S4 dispatch emulation.

6. **Lazy computation**: Credible sets, purity, summaries computed on access,
   not during fitting (unless needed for convergence).

---

## 4. Public API

### 4.1 Main Class: `SuSiE`

```python
class SuSiE:
    """Sum of Single Effects regression for Bayesian variable selection.

    Parameters
    ----------
    n_effects : int, default=10
        Maximum number of single effects (L). Automatically reduced to
        min(n_effects, n_variables) during fitting.

    prior_variance : float or array-like, default=0.2
        Prior variance for each effect, specified as a fraction of var(y).
        Internally transformed to absolute scale as V = prior_variance * var(y)
        for individual data, V = prior_variance * var_y (or yty/(n-1) if
        var_y not provided) for sufficient statistics, and
        V = prior_variance * var_y for summary statistics.
        See Section 5.3 (Scale Conventions) for details.

    estimate_prior_variance : bool, default=True
        If True, estimate the prior variance for each effect by empirical
        Bayes (maximizing the marginal likelihood).

    prior_variance_method : {'optim', 'em', 'simple'}, default='optim'
        Method for estimating prior variance.
        - 'optim': Brent's method on log(V), most accurate
        - 'em': EM update, fastest
        - 'simple': Compare V=V_init vs V=0

    estimate_residual_variance : bool, default=True
        If True, estimate the residual variance sigma^2 from data.
        This default applies to fit() and fit_from_sufficient_stats().
        fit_from_summary_stats() has its own estimate_residual_variance
        parameter that defaults to False independently (see its docstring).

    residual_variance_method : {'mom', 'mle'}, default='mom'
        Method for estimating residual variance.
        - 'mom': Method of moments
        - 'mle': Maximum likelihood

    prior_weights : array-like of shape (n_variables,), default=None
        Prior probability of each variable being causal. If None, uniform
        prior is used.

    null_weight : float, default=0.0
        Per-effect prior probability that the effect is absent. Must be
        in [0, 1). Each of the L single effects independently has
        probability null_weight of selecting the null variable (i.e.,
        contributing zero to the model). The implied prior probability
        that ALL L effects are absent is null_weight^L, not null_weight.

        Implementation: if > 0, an extra "null" variable (index p) is
        added internally as a virtual column: d and Xty are extended
        with a trailing zero, but X itself is not physically modified
        (see _FitData.has_null_column). For the SS/RSS path, XtX gains
        a zero row/column.

        Prior weight normalization: The user-supplied prior_weights (length
        p, or uniform if None) are rescaled to sum to (1 - null_weight),
        and the null column receives weight null_weight. Formally:
            pi[0:p] = prior_weights / sum(prior_weights) * (1 - null_weight)
            pi[p]   = null_weight

        SER null-column guard: when d[j]=0, the log Bayes factor is set
        to 0 (BF=1) and posterior moments are set to zero, so the null
        column's inclusion probability is determined entirely by its prior
        weight vs the data evidence for real variables.

        Result stripping: The null column (index p) is removed from all
        returned arrays (alpha, mu, mu2, pip, coef, lbf_variable,
        prior_weights) before constructing SuSiEResult. The null column's
        posterior mass is preserved separately as alpha_null (L-vector)
        in SuSiEResult, so that posterior_samples() can correctly
        represent "no effect for component l". The returned arrays always
        have shape consistent with the user's original p variables.
        Credible sets never include the null index. The returned
        prior_weights are NOT renormalized after stripping — they sum
        to (1 - null_weight), preserving the actual prior mass assigned
        to real variables.

    standardize : bool, default=True
        If True, center and scale columns of X to have mean 0 and
        variance 1 before fitting.

    intercept : bool, default=True
        If True, fit an intercept term.

    max_iter : int, default=100
        Maximum number of IBSS iterations.

    tol : float, default=1e-3
        Convergence tolerance. Algorithm stops when the change in ELBO
        (or max PIP change) is below this threshold.

    convergence_criterion : {'elbo', 'pip'}, default='elbo'
        What quantity to monitor for convergence.

    coverage : float, default=0.95
        Target coverage for credible sets.

    min_abs_corr : float, default=0.5
        Minimum absolute correlation (purity) for a credible set to be
        reported. Sets with lower purity are filtered out.

    refine : bool, default=False
        If True, run a post-fitting refinement step that attempts to
        escape local optima by re-fitting with modified prior weights.

    verbose : bool, default=False
        If True, print progress information during fitting.
    """

    def fit(self, X, y, **kwargs) -> 'SuSiE':
        """Fit SuSiE model with individual-level data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_variables)
            Predictor matrix. Can be dense numpy array, scipy sparse matrix,
            or pandas DataFrame.
        y : array-like of shape (n_samples,)
            Response vector.

        Returns
        -------
        self
            Returns self for sklearn Pipeline compatibility. Access full
            results via self.result_.
        """

    def fit_from_sufficient_stats(
        self, XtX, Xty, yty, n, *,
        X_col_means=None, y_mean=None,
        maf=None, var_y=None,
    ) -> 'SuSiE':
        """Fit SuSiE model from precomputed sufficient statistics.

        Parameters
        ----------
        XtX : array-like of shape (p, p)
            X'X matrix (after centering X if intercept is fitted).
        Xty : array-like of shape (p,)
            X'y vector (after centering).
        yty : float
            y'y scalar (after centering).
        n : int
            Number of samples.
        X_col_means : array-like of shape (p,), optional
            Column means of X before centering. Required if
            self.intercept=True; raises ValueError if missing and
            intercept is requested. If self.intercept=False, ignored.
        y_mean : float, optional
            Mean of y before centering. Same requirement as X_col_means:
            required when self.intercept=True, raises ValueError if
            missing. If self.intercept=False, ignored (intercept is 0.0).
        maf : array-like of shape (p,), optional
            Minor allele frequencies for filtering.
        var_y : float, optional
            Sample variance of y. If provided, used as the reference
            variance for scaling prior_variance (internal V = prior_variance
            * var_y). If None, the reference variance is computed as
            yty / (n-1) (ddof=1). See Section 5.3 (Scale Conventions).

        Returns
        -------
        self
        """

    def fit_from_summary_stats(
        self, *, z=None, R=None, n=None,
        bhat=None, shat=None,
        var_y=1.0, regularize_ld=0.0,
        estimate_residual_variance=None,
    ) -> 'SuSiE':
        """Fit SuSiE model from GWAS summary statistics.

        Provide EITHER z-scores OR (bhat, shat). Must always provide R.

        Parameters
        ----------
        z : array-like of shape (p,), optional
            Marginal z-scores (bhat / shat).
        R : array-like of shape (p, p)
            LD (correlation) matrix. Must be correlation, not r^2 or |r|.
        n : int, optional
            Sample size. If provided, enables PVE-adjustment of z-scores
            and residual variance estimation.
        bhat : array-like of shape (p,), optional
            Marginal effect size estimates.
        shat : array-like of shape (p,), optional
            Standard errors of marginal effect sizes.
        var_y : float, default=1.0
            Phenotype variance.
        regularize_ld : float | Literal['auto'], default=0.0
            LD matrix regularization parameter lambda.
            R_tilde = (1-lambda)*R + lambda*I.
            If 'auto', estimate lambda by maximum likelihood.
            Requires n to be provided (z_adj depends on n).
        estimate_residual_variance : bool, optional
            Override the estimator's estimate_residual_variance setting
            for this fit. If None (default), uses False — matching R's
            susie_rss() default convention. The rationale: summary-stat
            inputs rarely have exact sufficient statistics, so residual
            variance estimation can be unreliable. Pass True explicitly
            to enable estimation (requires n to be provided).

        Returns
        -------
        self
        """

    def predict(self, X) -> np.ndarray:
        """Predict responses for new data.

        Computes X @ coef_ + intercept_.

        Only supported after fit() or fit_from_sufficient_stats() when
        intercept metadata (X_col_means, y_mean) was available. Raises
        NotFittedError if called before fitting. Raises ValueError if
        called after fit_from_summary_stats(), because the summary-stats
        path does not retain enough preprocessing state (X_scale,
        X_center) to guarantee correct back-transformation of
        coefficients to the original feature scale. Users who need
        predictions from RSS results should manually compute
        X_new @ result_.coef + result_.intercept, understanding that
        coef is on the standardized scale.
        """

    @property
    def result_(self) -> 'SuSiEResult':
        """The fitted result (available after calling fit)."""
```

### 4.2 Result Object: `SuSiEResult`

```python
@dataclass(frozen=True)
class SuSiEResult:
    # Note: frozen=True prevents attribute reassignment, but NumPy arrays
    # are still mutable by default. The __post_init__ method sets
    # arr.flags.writeable = False on all stored arrays to enforce true
    # immutability. This prevents accidental modification of result data.
    # Core posterior quantities
    alpha: np.ndarray          # (L, p) posterior inclusion probabilities
    mu: np.ndarray             # (L, p) posterior means
    mu2: np.ndarray            # (L, p) posterior second moments
    prior_variance: np.ndarray # (L,) estimated prior variances V
    residual_variance: float   # estimated sigma^2
    prior_weights: np.ndarray  # (p,) prior inclusion weights (null column stripped if present)

    # Convergence
    elbo: np.ndarray           # ELBO at each iteration
    n_iter: int                # iterations run
    converged: bool            # whether algorithm converged

    # Regression coefficients
    coef: np.ndarray           # (p,) posterior mean coefficients
    intercept: float           # fitted intercept (or 0.0)

    # Log Bayes factors
    lbf: np.ndarray            # (L,) per-effect log BF
    lbf_variable: np.ndarray   # (L, p) per-variable log BF

    # Metadata
    n_samples: int
    n_variables: int
    feature_names: list[str] | None

    # Null-effect tracking (only present when null_weight > 0)
    # alpha_null[l] = posterior probability that effect l is null (no variable)
    # This is the stripped null column's mass, needed for posterior_samples()
    # to correctly represent "no effect for component l". When null_weight=0,
    # this is None.
    alpha_null: np.ndarray | None  # (L,) or None

    @cached_property
    def pip(self) -> np.ndarray:
        """Posterior inclusion probabilities (p-vector).

        PIP_j = 1 - prod_l (1 - alpha[l,j])
        Computed in log space for numerical stability.
        """

    def get_credible_sets(
        self, X=None, R=None, *,
        coverage=0.95, min_abs_corr=0.5,
        n_purity=100,
    ) -> list['CredibleSet']:
        """Extract credible sets with purity filtering.

        Parameters
        ----------
        X : array-like, optional
            Original data matrix (for computing purity via correlations).
        R : array-like, optional
            Correlation matrix (alternative to X for purity).
        coverage : float
            Target coverage level.
        min_abs_corr : float
            Minimum purity threshold.
        n_purity : int
            Maximum CS size for purity subsampling.

        Returns
        -------
        list of CredibleSet
        """

    def posterior_mean(self, prior_tol=1e-9) -> np.ndarray:
        """E[b] = sum_l alpha[l,:] * mu[l,:], excluding near-zero effects."""

    def posterior_sd(self, prior_tol=1e-9) -> np.ndarray:
        """Standard deviation of the posterior on each coefficient."""

    def posterior_samples(self, n_samples=100, *, rng=None) -> np.ndarray:
        """Draw samples from the approximate posterior.

        For each sample and each effect l:
          1. Draw gamma_l from Categorical(alpha[l,:]) — or "null" with
             probability alpha_null[l] if null_weight was used
          2. If null: b_l = 0. Else: draw b_l[gamma_l] ~ N(mu[l,gamma_l], var)
        Sum over effects to get b.

        Returns array of shape (n_samples, p).
        """

    def lsfr(self) -> np.ndarray:
        """Local false sign rate for each variable."""

    def summary(self) -> 'pd.DataFrame':
        """Summary table with credible sets, PIPs, and effect sizes."""

    def plot(self, *, y_type='pip', ax=None, **kwargs):
        """Plot PIPs or z-scores with credible sets highlighted."""
```

### 4.3 Credible Set Object

```python
@dataclass(frozen=True)
class CredibleSet:
    variables: np.ndarray      # indices of variables in this set
    coverage: float            # achieved coverage
    effect_index: int          # which of the L effects this corresponds to
    log_bayes_factor: float    # lbf for this effect
    purity: PurityMetrics | None  # None if purity not computed

@dataclass(frozen=True)
class PurityMetrics:
    min_abs_corr: float
    mean_abs_corr: float
    median_abs_corr: float
```

### 4.4 Convenience Functions

```python
def compute_sufficient_stats(X, y, *, standardize=True):
    """Compute X'X, X'y, y'y from individual-level data.

    Returns dict with keys: XtX, Xty, yty, n, X_col_means, y_mean
    """

def univariate_regression(X, y, *, center=True, scale=False):
    """Fit marginal univariate regressions (one per column of X).

    Returns dict with keys: betahat, sebetahat, z_scores
    """

def estimate_ld_regularization(z, R, n=None):
    """Estimate optimal LD regularization parameter lambda.

    Returns float lambda in [0, 1].
    """
```

### 4.5 Auto-Fitting

```python
def susie_auto(X, y, *, L_init=1, L_max=512, **kwargs) -> SuSiEResult:
    """Fit SuSiE with automatic selection of L.

    Starts with L_init effects and doubles L until no new effects
    have non-zero prior variance, or L_max is reached.

    Uses a multi-stage fitting procedure:
      Stage 1: Fixed residual variance, fixed prior variance
      Stage 2: Estimate residual variance, fixed prior variance
      Stage 3: Estimate both (production fit)
    """
```

---

## 5. Internal Data Representation

### 5.1 Preprocessed Data

All three input paths (individual, sufficient stats, summary stats) are
converted to a common internal representation before the IBSS loop:

```python
@dataclass
class _FitData:
    """Internal data representation for the IBSS algorithm."""

    # Sufficient statistics (always available after preprocessing)
    XtX: np.ndarray | None     # (p, p) - None if individual data uses lazy products
    Xty: np.ndarray            # (p,) X'y
    yty: float                 # y'y
    d: np.ndarray              # (p,) diagonal of X'X
    n: int                     # sample size
    p: int                     # number of variables

    # Individual-level data (optional, for matrix-vector products)
    # For dense data: X is the centered/scaled matrix.
    # For sparse data: X is the RAW sparse matrix, and centering/scaling
    # is applied implicitly via X_center and X_scale in compute_Xb/compute_Xty.
    # This avoids densifying sparse matrices.
    X: np.ndarray | scipy.sparse.spmatrix | None  # (n, p) or None for SS/RSS
    y: np.ndarray | None       # (n,) centered, or None for SS/RSS

    # Preprocessing metadata (used for implicit centering of sparse X)
    X_center: np.ndarray | None   # (p,) column means before centering
    X_scale: np.ndarray | None    # (p,) column SDs before scaling
    y_mean: float                 # mean of y before centering
    is_sparse: bool               # True if X is a sparse matrix

    # Null column tracking
    has_null_column: bool              # True if null_weight > 0
    # When True, d and Xty have length p (including the virtual null at index p-1)
    # but X has only p-1 physical columns. compute_Xb/compute_Xty handle this:
    # the null column contributes 0 to all products, so its entry is skipped.

    # For RSS-lambda variant
    eigen_values: np.ndarray | None    # eigenvalues of R
    eigen_vectors: np.ndarray | None   # eigenvectors of R
    regularization: float              # lambda

    def compute_Xb(self, b: np.ndarray) -> np.ndarray:
        """Compute (centered/scaled X) @ b.

        b has length self.p (which includes the virtual null column if
        has_null_column=True). When has_null_column=True, the last
        element of b is ignored (the null column contributes zero).
        Let b_phys = b[:-1] if has_null_column else b.

        For dense X: returns X @ b_phys (X already transformed).
        For sparse X: returns X @ (b_phys / scale) - sum(center * b_phys / scale),
            applying centering and scaling implicitly without densifying.
        For SS/RSS (no X): raises — use compute_XtXb instead.
        """

    def compute_Xty(self, r: np.ndarray) -> np.ndarray:
        """Compute (centered/scaled X)' @ r.

        Returns a self.p-length vector. When has_null_column=True, the
        last element is always 0.0 (the null column's inner product with
        any vector is zero).

        For dense X: result[:-1] = X.T @ r; result[-1] = 0.0.
        For sparse X: result[:-1] = (X.T @ r) / scale - (center / scale) * sum(r);
            result[-1] = 0.0.
        When has_null_column=False, the trailing zero is not appended.
        """

    def compute_XtXb(self, b: np.ndarray) -> np.ndarray:
        """Compute X'X @ b using XtX if available, else X.T @ (X @ b).

        When has_null_column=True and XtX is None (individual data),
        computes X.T @ (X @ b[:-1]) and appends 0.0. When XtX is
        available, XtX already includes the null row/column of zeros.
        """
```

### 5.2 Model State (Mutable During Fitting)

```python
@dataclass
class _ModelState:
    """Mutable state tracked during IBSS iterations."""

    alpha: np.ndarray       # (L, p)
    mu: np.ndarray          # (L, p)
    mu2: np.ndarray         # (L, p)
    V: np.ndarray           # (L,) prior variances
    sigma2: float           # residual variance
    KL: np.ndarray          # (L,) KL divergence per effect
    lbf: np.ndarray         # (L,) log Bayes factor per effect
    lbf_variable: np.ndarray  # (L, p)

    # Fitted value caches — exactly one of Xr/XtXr is non-None per fit.
    # Individual-data path: Xr = X @ E[b], an n-vector.
    # SS/RSS path: XtXr = X'X @ E[b], a p-vector.
    # These are maintained incrementally during IBSS iterations and
    # also used to compute the ELBO efficiently (see Section 7.3).
    Xr: np.ndarray | None          # (n,) individual-data path only
    XtXr: np.ndarray | None        # (p,) sufficient-stats/RSS path only

    # Per-effect norm caches for efficient ELBO computation.
    # Xb_sq_norms[l] = ||X @ b_l||^2  (individual) or b_l @ XtX @ b_l (SS)
    # where b_l = alpha[l,:] * mu[l,:].
    # Updated as a byproduct of the IBSS residual update (see Section 7.2).
    Xb_sq_norms: np.ndarray        # (L,) per-effect squared norms

    # For unmappable effects
    theta: np.ndarray | None       # (p,) background effects
    tau2: float | None             # background variance
    ash_pi: np.ndarray | None      # Mr.ASH mixture weights
```

### 5.3 Scale Conventions

The `prior_variance` parameter is specified by the user as a fraction of
phenotype variance (default 0.2). Each entry point transforms this to an
absolute scale before passing it to the IBSS loop. The IBSS loop and all
internal functions (`fit_ser`, `compute_elbo`, etc.) always work in
**absolute variance units**.

| Entry point | User-specified V | Reference variance | Internal V (absolute) |
|---|---|---|---|
| `fit(X, y)` | fraction of ref | `var(y)` after centering | `V * var(y)` |
| `fit_from_sufficient_stats(...)` | fraction of ref | `var_y` if provided, else `yty / (n-1)` | `V * ref` |
| `fit_from_summary_stats(...)` | fraction of ref | `var_y` parameter (default 1.0) | `V * var_y` |

In all cases: `internal_V = prior_variance * reference_variance`. The
reference variance is the best available estimate of var(y).

**Degrees of freedom convention (ddof)**:
- `fit(X, y)`: `var(y)` uses ddof=1 (i.e., `yty / (n-1)` where yty is
  computed after centering). This matches `np.var(y, ddof=1)`.
- `fit_from_sufficient_stats(...)`: The caller provides `yty` and `n`.
  If `var_y` is not provided, the reference is `yty / (n-1)` (ddof=1).
- `fit_from_summary_stats(...)`: The RSS mapping uses `(n-1)` consistently:
  `XtX = (n-1) * R`, `Xty = sqrt(n-1) * z_adj`, `yty = n-1`. This means
  `yty / (n-1) = 1`, so when `var_y=1.0` (default), all scales are
  self-consistent.

**Numerical equivalence**:
- `fit(X, y)` and `fit_from_sufficient_stats(X'X, X'y, y'y, n)` should
  produce identical results to floating-point tolerance, provided the
  sufficient statistics are computed after the same centering/scaling.
- `fit_from_summary_stats(z, R, n)` should produce numerically close
  results to the above under these strict assumptions: R is the exact
  in-sample correlation matrix (`np.corrcoef(X.T)` after centering/scaling),
  z-scores are derived from the same X and y, `regularize_ld=0.0`,
  and `var_y` matches the actual sample variance. In practice, external
  LD matrices, LD regularization, or PVE-adjustment introduce small
  discrepancies.

**Back-transformation for reporting**: When `X_scale` is available (i.e.,
after `fit()` or `fit_from_sufficient_stats()` with standardization),
coefficients and intercept are transformed back to the original
(un-standardized) scale when constructing the result. Specifically:
- `coef[j]` is divided by `X_scale[j]` to return to the original X scale
- `intercept` is adjusted for centering: `y_mean - X_center @ coef`

After `fit_from_summary_stats()`, no per-column scaling factors are
available, so `coef` remains on the internal (standardized) scale. Users
who need predictions from RSS results should be aware of this (see
`predict()` docstring above).

In all cases, `prior_variance[l]` remains on the internal (standardized)
scale in the result, matching R's convention; users who need the original
scale can multiply by the column-variance ratios.

**Residual variance**: `sigma2` (residual variance) is always on the scale
of the response y after centering. For RSS with `var_y=1.0`, `sigma2` is
a proportion of phenotype variance.

---

## 6. Performance Analysis and Optimization Strategy

### 6.1 Computational Complexity

| Operation | Complexity | Frequency | Total per iteration |
|-----------|-----------|-----------|-------------------|
| SER: X'r (individual data) | O(np) | L times | O(npL) |
| SER: XtX @ b (sufficient stats) | O(p^2) | L times | O(p^2 L) |
| SER: Bayes factor computation | O(p) | L times | O(pL) |
| SER: Softmax (posterior weights) | O(p) | L times | O(pL) |
| Prior variance optimization | O(p) | L times | O(pL) |
| ELBO (with Xr/XtXr + Xb_sq_norms caches) | O(n + pL) or O(p + pL) | 1 time | O(n + pL) or O(p + pL) |
| ELBO (R's approach, recompute per-effect products) | O(npL) or O(p^2 L) | 1 time | O(npL) or O(p^2 L) |
| Residual variance estimation | O(pL) | 1 time | O(pL) |

Note: The ELBO requires `sum_l b_l' X'X b_l` (where b_l = E[b_l]), which
uses the FULL X'X matrix — not just the diagonal. The R implementation
recomputes L matrix-vector products each time, making its ELBO as expensive
as the SER updates themselves. Our implementation avoids this by caching
`Xb_sq_norms[l] = ||X @ b_l||^2` (or `b_l @ XtX @ b_l`) as a byproduct
of the IBSS residual update (see Section 7.2), making the ELBO itself
O(n + pL) or O(p + pL). The total per-iteration cost is dominated by the
SER updates: O(npL) for individual data, O(p^2 L) for SS.

**Dominant cost**: Matrix-vector products in SER:
- Individual data: O(npL) per iteration  (n=samples, p=variables, L=effects)
- Sufficient stats: O(p^2 L) per iteration

For typical genetics problems: n ~ 500-50,000, p ~ 200-50,000, L ~ 10.

### 6.2 Performance Bottleneck Identification

**Bottleneck 1: SER matrix-vector products** (60-80% of runtime)

For individual data, computing `X' @ r` for each of L effects is the dominant
cost. When X is stored and n > p, this is O(np) per SER call.

*Strategy*: Use BLAS-backed `np.dot()`. Column-major storage (Fortran order)
for X gives optimal cache behavior for column access. For sparse X, use
`scipy.sparse` matrix-vector products.

**Bottleneck 2: Residual updates** (10-20% of runtime)

After each SER update, the "expected residuals" must be updated. With
individual data, this requires computing X @ (alpha_new * mu_new - alpha_old * mu_old).

*Strategy*: Track `Xr = X @ E[b]` incrementally. Each SER update changes only
one row of alpha/mu, so the update is a rank-1 correction: `Xr += X @ delta_b`
where `delta_b` is sparse (nonzero only where alpha changed significantly).
In practice, the full vector update `X @ delta_b_l` is O(np) but with a small
constant.

**Bottleneck 3: ELBO computation** (5-10% of runtime)

The ELBO involves summing over L effects and p variables. The cross-term
`E[b' X'X b]` requires `sum_l b_l' X'X b_l`, which uses the full X'X
matrix and is O(npL) or O(p^2 L) if computed from scratch (as the R
implementation does).

*Strategy*: Cache `Xb_sq_norms[l] = ||X @ b_l||^2` (or `b_l @ XtX @ b_l`)
as a byproduct of the IBSS residual update — the matrix-vector product is
already computed to update Xr/XtXr, so the norm is just an O(n) or O(p)
dot product. Combined with cached Xr/XtXr, the ELBO itself then costs only
O(n + pL) or O(p + pL). See Section 7.3 for the full derivation.

**Bottleneck 4: Mr.ASH coordinate ascent** (only for SuSiE-ash)

The Mr.ASH inner loop is inherently sequential (each coefficient update
depends on the previous residual). This is the one operation where Python
loop overhead matters significantly.

*Strategy*: Implement the inner loop with Numba JIT compilation. The
`@numba.njit` decorator eliminates Python overhead and achieves 95%+ of C++
performance. Fallback to pure NumPy for environments without Numba.

### 6.3 Implementation Tiers

**Tier 1: Pure NumPy/SciPy** (always available)
- All core SuSiE operations: SER, IBSS, ELBO, credible sets
- Matrix products via np.dot (BLAS-backed)
- Optimization via scipy.optimize.minimize_scalar
- Expected performance: within 2-3x of R/C++ for SuSiE core

**Tier 2: Numba-accelerated** (optional dependency)
- Mr.ASH coordinate ascent loop
- SER inner loop for very large p
- Expected performance: within 1.1x of R/C++

**Tier 3: Sparse matrix support** (scipy.sparse)
- For genotype matrices with structure (e.g., trend filtering)
- Sparse X'X computation and storage
- Expected performance: significant speedup for sparse problems

### 6.4 Memory Optimization

| Object | Size | Strategy |
|--------|------|----------|
| X (individual) | n*p*8 bytes | Store as Fortran-order for column access |
| XtX (sufficient stats) | p*p*8 bytes | Symmetric: use scipy.linalg functions |
| alpha, mu, mu2 | 3*L*p*8 bytes | Always needed; L typically small |
| Xr (residual tracker) | n*8 bytes | Single vector, updated in-place |
| R (LD matrix) | p*p*8 bytes | Consider sparse if banded |

For a typical fine-mapping problem (n=5000, p=1000, L=10):
- X: 40 MB
- XtX: 8 MB
- Posterior matrices: 0.24 MB
- Total: ~50 MB (very manageable)

For large-scale (n=50000, p=50000, L=10):
- X: 20 GB (must use sufficient stats path)
- XtX: 20 GB (consider sparse/low-rank)
- Posterior matrices: 12 MB
- Individual-level data fitting not feasible; use summary stats

### 6.5 Numerical Stability Plan

1. **Log-sum-exp everywhere**: Posterior weight computation uses
   `scipy.special.logsumexp` to prevent overflow/underflow.

2. **PIP in log space**: `log(1 - PIP_j) = sum_l log(1 - alpha[l,j])` avoids
   underflow when many effects contribute.

3. **Eigenvalue truncation**: For LD matrix eigendecomposition, set negative
   eigenvalues to a small positive floor (e.g., 1e-10).

4. **Safe correlation computation**: Handle constant columns (zero variance)
   gracefully when computing purity.

5. **Tolerance constants**:
   - `prior_tol = 1e-9`: threshold below which prior variance is treated as zero
   - `convergence_tol = 1e-3`: default ELBO convergence threshold
   - `purity_tol = 0.5`: default minimum absolute correlation for CS
   - `elbo_decrease_tol = 1e-6`: warning threshold for ELBO decrease

---

## 7. Module Specifications

### 7.1 `_ser.py` — Single Effect Regression

The SER is the fundamental building block. It fits a single-effect model to
residuals, returning posterior quantities.

```python
def fit_ser(
    Xty_residual: np.ndarray,   # (p,) X' @ residual
    d: np.ndarray,              # (p,) diagonal of X'X
    sigma2: float,              # residual variance
    prior_variance: float,      # V_l
    prior_weights: np.ndarray,  # (p,) prior inclusion probabilities (linear scale, sum to 1)
    estimate_prior_variance: bool,
    prior_variance_method: str,
    check_null_threshold: float,
) -> SERResult:
    """Fit single-effect regression model.

    Core computation:
        betahat_j = Xty_residual[j] / d[j]
        shat2_j = sigma2 / d[j]
        lbf_j = -0.5 * log(1 + V/shat2_j) + 0.5 * betahat_j^2 * V / (shat2_j * (shat2_j + V))
        alpha = softmax(lbf + log(prior_weights))
        mu_j = V / (V + shat2_j) * betahat_j
        mu2_j = sigma2 * V / (sigma2 + V * d[j]) + mu_j^2

    Null-column guard: When null_weight > 0, the last column has d[j]=0.
    For any j where d[j]==0: set lbf[j]=0, mu[j]=0, mu2[j]=0. This
    ensures the null column's posterior weight is determined entirely by
    its prior weight (no division by zero).

    Returns
    -------
    SERResult with: alpha, mu, mu2, lbf, lbf_variable, V, KL
    """
```

**Prior variance estimation** (private function in `_ser.py`, called within `fit_ser`):

```python
def _optimize_prior_variance(
    betahat: np.ndarray,     # (p,)
    shat2: np.ndarray,       # (p,)
    prior_weights: np.ndarray,  # (p,) linear-scale probabilities (same as fit_ser)
    V_init: float,
    method: str,             # 'optim', 'em', 'simple'
) -> float:
    """Estimate prior variance V by empirical Bayes.

    'optim': Maximize the log marginal likelihood of the SER model,
             log(sum_j pi_j * BF_j(V)), over V using Brent's method.
             This is the log of a weighted sum of Bayes factors (NOT a
             sum of logs). The weights pi_j are the prior inclusion
             probabilities. Optimization is on log(V) with bounds
             [-30, 15].
    'em':    V_new = sum(alpha * mu2) / sum(alpha)
    'simple': Compare V=V_init vs V=0, keep better.
    """
```

### 7.2 `_ibss.py` — IBSS Algorithm

```python
def ibss_loop(
    data: _FitData,
    state: _ModelState,
    params: dict,
) -> tuple[_ModelState, list[float]]:
    """Run the IBSS coordinate ascent loop.

    For each iteration:
      For l in range(L):
        1. Compute X'r_l (X-transposed residual for effect l):
           - Individual-data path (maintains Xr, an n-vector):
               Xr -= X @ (alpha[l,:] * mu[l,:])   # remove effect l
               X'r_l = X' @ (y - Xr)              # full matrix-vector product
           - SS/RSS path (maintains XtXr, a p-vector):
               XtXr -= XtX @ (alpha[l,:] * mu[l,:])  # remove effect l
               X'r_l = Xty - XtXr                    # p-vector subtraction

        2. Call fit_ser(X'r_l, d, ...) to get updated alpha[l], mu[l], mu2[l], V[l]

        3. Add back updated effect l:
           - Individual: Xr += X @ (alpha[l,:] * mu[l,:])
           - SS/RSS: XtXr += XtX @ (alpha[l,:] * mu[l,:])

      Compute ELBO
      Optionally update sigma2
      Check convergence

    Returns updated state and ELBO trace.
    """
```

**Residual computation strategy**:

For individual data (Xr is an n-vector = X @ E[b]):
```
Maintain: Xr = X @ sum_l (alpha[l,:] * mu[l,:])
Before SER l:
  b_l = alpha[l,:] * mu[l,:]                  # (p,) current expected effect
  Xr -= X @ b_l                               # remove effect l from Xr
  X'r_l = X' @ (y - Xr)                       # full matrix-vector product (O(np))
After SER l:
  b_l_new = alpha[l,:] * mu[l,:]              # updated
  Xb_l = X @ b_l_new                          # (n,) already needed for Xr update
  Xr += Xb_l                                  # add back updated effect
  Xb_sq_norms[l] = Xb_l @ Xb_l               # cache for ELBO (O(n) dot product)
```
Note: The `X' @ (y - Xr)` computation MUST use a full matrix-vector product.
Even when X is standardized, X'X is NOT diagonal, so there is no shortcut
like `d * b_l`. The R implementation uses `compute_Xb(data$X, ...)` which
performs the full product.

For sufficient stats / RSS (XtXr is a p-vector = X'X @ E[b]):
```
Maintain: XtXr = XtX @ sum_l (alpha[l,:] * mu[l,:])
Before SER l:
  b_l = alpha[l,:] * mu[l,:]
  XtXr -= XtX @ b_l                           # remove effect l (O(p^2))
  X'r_l = Xty - XtXr                          # simple subtraction (O(p))
After SER l:
  b_l_new = alpha[l,:] * mu[l,:]
  XtXb_l = XtX @ b_l_new                      # (p,) already needed for XtXr update
  XtXr += XtXb_l                              # add back updated effect
  Xb_sq_norms[l] = b_l_new @ XtXb_l           # cache for ELBO (O(p) dot product)
```

### 7.3 `_elbo.py` — ELBO Computation

```python
def compute_elbo(
    data: _FitData,
    state: _ModelState,
) -> float:
    """Compute the Evidence Lower Bound.

    ELBO = expected_log_likelihood - sum(KL)

    expected_log_likelihood:
        = -n/2 * log(2*pi*sigma2) - 1/(2*sigma2) * E[||y - Xb||^2]

    E[||y - Xb||^2]:
        = yty
        - 2 * sum_j (sum_l alpha[l,j]*mu[l,j]) * Xty[j]
        + sum_j d[j] * sum_l alpha[l,j]*mu2[l,j]
        + sum_{l!=l'} sum_j sum_k alpha[l,j]*mu[l,j] * XtX[j,k] * alpha[l',k]*mu[l',k]

    Expanding E[b' X'X b] under the variational posterior:
        E[b' X'X b] = sum_l E[b_l' X'X b_l]
                     + sum_{l!=l'} E[b_l]' X'X E[b_{l'}]

    Where (since gamma_l is one-hot under the SER posterior):
        E[b_l' X'X b_l] = sum_j alpha[l,j] * mu2[l,j] * d[j]   # diagonal only, O(pL)

    And letting b_l = E[b_l] = alpha[l,:] * mu[l,:]:
        sum_{l!=l'} b_l' X'X b_{l'} = b_bar' X'X b_bar - sum_l b_l' X'X b_l

    The cross-effect term sum_l b_l' X'X b_l uses the FULL matrix X'X
    (not just the diagonal), because b_l = alpha[l,:]*mu[l,:] is a dense
    p-vector. The R implementation recomputes this from scratch each time
    (O(npL) for individual data, O(p^2 L) for SS), but we can do better.

    Efficient implementation using per-effect norm caches:

    During the IBSS loop, we already compute X @ b_l (to update Xr) or
    XtX @ b_l (to update XtXr). As a byproduct, we cache:
        Xb_sq_norms[l] = ||X @ b_l||^2     # individual: O(n) from dot product
        Xb_sq_norms[l] = b_l @ (XtX @ b_l) # SS: O(p) from dot product

    Then the full E[RSS] formula becomes:

    Individual-data path:
        E[RSS] = ||y - Xr||^2                          # O(n), uses cached Xr
               - sum_l Xb_sq_norms[l]                   # O(L), cached scalars
               + sum_j d[j] * sum_l alpha[l,j]*mu2[l,j] # O(pL), diagonal only

    SS/RSS path:
        E[RSS] = yty - 2 * b_bar @ Xty                 # O(p)
               + b_bar @ XtXr                           # O(p), uses cached XtXr
               - sum_l Xb_sq_norms[l]                   # O(L), cached scalars
               + sum_j d[j] * sum_l alpha[l,j]*mu2[l,j] # O(pL), diagonal only

    Total ELBO cost: O(n + pL) for individual data, O(p + pL) for SS/RSS.
    This requires the Xb_sq_norms cache (L scalars) in _ModelState, updated
    during the IBSS loop as a byproduct of the residual update.
    """
```

### 7.4 `_credible_sets.py` — Credible Set Extraction

```python
def extract_credible_sets(
    alpha: np.ndarray,           # (L, p)
    prior_variance: np.ndarray,  # (L,)
    coverage: float = 0.95,
    prior_tol: float = 1e-9,
) -> list[tuple[np.ndarray, float, int]]:
    """Extract raw credible sets from posterior inclusion probabilities.

    For each effect l with V[l] > prior_tol:
      1. Sort alpha[l,:] in decreasing order
      2. Cumsum until >= coverage
      3. Return (variable_indices, achieved_coverage, l)

    Deduplicate: when two effects l1 and l2 produce identical variable
    sets (same indices), keep only the one with the higher log Bayes
    factor. The retained CredibleSet preserves its original effect_index
    and log_bayes_factor; the duplicate is discarded. This can happen
    when multiple effects converge to the same signal.
    """


def compute_purity(
    cs_variables: np.ndarray,
    X: np.ndarray | None = None,
    R: np.ndarray | None = None,
    n_purity: int = 100,
) -> PurityMetrics:
    """Compute purity (min/mean/median absolute correlation) for a credible set.

    If len(cs_variables) > n_purity, subsample to n_purity variables.
    Compute pairwise correlations from X or R.
    """
```

### 7.5 `_preprocessing.py` — Data Validation and Transformation

```python
def preprocess_individual_data(
    X, y, *,
    standardize: bool = True,
    intercept: bool = True,
    null_weight: float = 0.0,
) -> _FitData:
    """Validate and preprocess individual-level data.

    Steps:
    1. Convert to numpy arrays (float64). Accept scipy.sparse matrices as-is.
    2. Check for NaN, constant columns
    3. Compute column means and (if standardize) column SDs
    4. Dense path:
       a. Center y (subtract mean) if intercept=True
       b. Center columns of X (subtract column means) if intercept=True
       c. Scale columns of X (divide by column SDs) if standardize=True
       d. Compute d = column sum-of-squares of the transformed X
       e. Compute Xty = X' @ y (using transformed X and centered y)
    5. Sparse path (X is scipy.sparse):
       a. Center y (subtract mean) if intercept=True
       b. Store X_center, X_scale as attributes on _FitData; do NOT
          modify X itself (centering would destroy sparsity)
       c. Compute d and Xty using implicit centering formulas:
          d[j] = (X[:,j]' @ X[:,j]) / scale[j]^2 - n * (center[j] / scale[j])^2
          Xty[j] = (X[:,j]' @ y) / scale[j] - center[j] / scale[j] * sum(y)
       d. All subsequent matrix-vector products go through
          _FitData.compute_Xb() / compute_Xty() which apply centering
          and scaling implicitly (see Section 5.1)
    6. If null_weight > 0, handle null column virtually:
       - Extend d with a trailing 0: d = np.append(d, 0.0)
       - Extend Xty with a trailing 0: Xty = np.append(Xty, 0.0)
       - Set _FitData.p = p + 1, _FitData.has_null_column = True
       - Do NOT physically append a column to X. The IBSS loop and SER
         never need X @ e_{p+1} (it would be zero), and the null column
         guard in fit_ser handles d[j]=0 directly. For sparse X, this
         avoids the cost of modifying the sparse structure.
    """


def preprocess_sufficient_stats(
    XtX, Xty, yty, n, *,
    standardize: bool = True,
    check_psd: bool = False,
    maf: np.ndarray | None = None,
) -> _FitData:
    """Validate and preprocess sufficient statistics.

    Steps:
    1. Symmetrize XtX: (XtX + XtX.T) / 2
    2. Filter by MAF if provided
    3. Standardize if requested — scales to unit variance (not unit norm):
       Let csd[j] = sqrt(XtX[j,j] / (n-1))  (column SD, ddof=1).
       a. XtX_std[j,k] = XtX[j,k] / (csd[j] * csd[k])
       b. Xty_std[j] = Xty[j] / csd[j]
       c. yty is unchanged (y is not rescaled)
       d. d_std[j] = XtX_std[j,j] = XtX[j,j] / csd[j]^2 = n-1
       e. Store csd as _FitData.X_scale for back-transformation:
          coef_original[j] = coef_std[j] / csd[j]
       This matches fit(X, y, standardize=True), where each column is
       divided by its sample SD (ddof=1), yielding d[j] = n-1. Using
       csd (not column norm) ensures d_std = n-1, not 1.
    4. If not standardizing: d = diag(XtX) directly.
    5. Check positive semi-definiteness if requested
    """


def preprocess_summary_stats(
    z, R, n, *,
    bhat=None, shat=None,
    var_y: float = 1.0,
    regularize_ld: float | Literal['auto'] = 0.0,
) -> _FitData:
    """Convert summary statistics to sufficient statistics form.

    Steps:
    1. Compute z from bhat/shat if not provided directly
    2. PVE-adjust z-scores if n is provided:
       adj = (n-1) / (z^2 + n - 2)
       z_adj = sqrt(adj) * z
    3. Construct sufficient statistics:
       XtX = (n-1) * R  (or R if n not provided)
       Xty = sqrt(n-1) * z_adj  (or z if n not provided)
       yty = n - 1  (or 1 if n not provided)
    4. Apply LD regularization:
       - If regularize_ld == 0.0: no regularization
       - If regularize_ld == 'auto': estimate lambda by maximizing
         N(z_adj; 0, (1-lambda)*R + lambda*I), then apply.
         Requires n to be provided (z_adj depends on n). Raises
         ValueError if n is None and regularize_ld='auto'.
       - If regularize_ld is a float > 0: use that value as lambda
         (does not require n)
       In all cases: R_tilde = (1 - lambda) * R + lambda * I
    5. Eigendecompose R_tilde (the regularized matrix) if using RSS-lambda path
    """
```

### 7.6 `_mrash.py` — Mr.ASH Implementation

```python
def fit_mrash(
    X: np.ndarray,           # (n, p)
    y: np.ndarray,           # (n,)
    sa2: np.ndarray,         # (K,) mixture component variances
    *,
    pi: np.ndarray | None = None,     # (K,) initial mixture proportions
    beta_init: np.ndarray | None = None,  # (p,) initial coefficients
    sigma2: float | None = None,
    max_iter: int = 100,
    min_iter: int = 5,
    tol: float = 1e-3,
    update_pi: bool = True,
    update_sigma2: bool = True,
) -> MrASHResult:
    """Fit Mr.ASH model via coordinate ascent variational EM.

    This is the performance-critical function that benefits from Numba
    acceleration. The implementation attempts to import numba and falls
    back to a pure-numpy loop if unavailable.
    """
```

Numba-accelerated kernel:

```python
# In _numba_kernels.py
try:
    import numba as nb

    @nb.njit(cache=True)
    def _mrash_loop(X, w, sa2, pi, beta, r, sigma2, order,
                    max_iter, min_iter, tol, eps,
                    update_pi, update_sigma2):
        """JIT-compiled Mr.ASH coordinate ascent loop.

        This is the inner loop that processes each variable in sequence,
        updating coefficients, residuals, and mixture proportions.

        Performance: achieves 95%+ of equivalent C++ code.
        """
        n, p = X.shape
        K = len(sa2)
        # ... (full loop as specified in C++ analysis)

except ImportError:
    # Fallback: pure numpy implementation (slower for Mr.ASH)
    def _mrash_loop(X, w, sa2, pi, beta, r, sigma2, order, ...):
        # Use vectorized numpy operations where possible
        # Accept ~60-70% of C++ performance
        ...
```

### 7.7 `_plotting.py` — Visualization

```python
def plot_pip(
    result: SuSiEResult,
    *,
    credible_sets: list[CredibleSet] | None = None,
    ax=None,
    highlight_cs: bool = True,
    add_legend: bool = True,
    colors: list | None = None,
    **kwargs,
):
    """Manhattan-style plot of posterior inclusion probabilities.

    Variables in credible sets are highlighted with distinct colors.
    """

def plot_diagnostic(
    result: SuSiEResult,
    *,
    what: str = 'elbo',  # 'elbo', 'alpha', 'prior_variance'
    ax=None,
):
    """Diagnostic plot for convergence assessment."""

def plot_changepoint(
    result: SuSiEResult,
    y: np.ndarray,
    *,
    ax=None,
):
    """Specialized plot for trend filtering results."""
```

### 7.8 `datasets.py` — Example Data

```python
def load_example(name='N3finemapping') -> dict:
    """Load a bundled example dataset.

    Available datasets:
    - 'N3finemapping': 574 samples, 1001 variables, 3 true causal effects
    - 'small': Small simulated dataset for quick testing

    Returns dict with keys: X, y, true_coef, [additional metadata]
    """
```

---

## 8. Implementation Plan

### Phase 1: Core SuSiE (Individual Data)

**Priority: HIGH — This is the minimum viable package.**

1. `_types.py`: `SuSiEResult`, `CredibleSet`, `PurityMetrics`, `_FitData`, `_ModelState`
2. `_preprocessing.py`: Data validation, centering/scaling, SS/RSS conversion
3. `_ser.py`: Single effect regression + prior variance estimation (optim, EM, simple)
4. `_elbo.py`: ELBO computation
5. `_ibss.py`: IBSS loop with convergence checking + residual variance estimation
6. `_credible_sets.py`: CS extraction and purity filtering
7. `susie.py`: `SuSiE` class with `fit()`, `predict()`, and result assembly
8. `datasets.py`: At least one example dataset
9. Tests: Compare outputs against R package on example data

**Key correctness checks:**
- ELBO is monotonically non-decreasing
- alpha rows sum to 1 (when null_weight=0); sum to <= 1 when null_weight > 0
  (the missing mass is the stripped null column's posterior probability)
- 0 <= PIP <= 1
- Credible sets achieve requested coverage
- Coefficients match R output within tolerance

### Phase 2: Summary Statistics Support

**Priority: HIGH — Critical for genetics users.**

1. Add `preprocess_sufficient_stats` and `preprocess_summary_stats` to
   `_preprocessing.py` (SS/RSS conversion lives here per module layout)
2. Add `fit_from_sufficient_stats()` and `fit_from_summary_stats()` to SuSiE
3. LD regularization (lambda estimation)
4. PVE adjustment of z-scores
5. Tests comparing `fit()` vs `fit_from_sufficient_stats()` for same data

### Phase 3: Refinement and Auto-Fitting

**Priority: MEDIUM**

1. CS refinement procedure (re-fit with modified prior weights)
2. `susie_auto()` function with adaptive L selection
3. Initialization from external coefficients

### Phase 4: SuSiE-ash / SuSiE-inf

**Priority: MEDIUM-LOW — Advanced feature for polygenic architectures.**

1. `_mrash.py`: Mr.ASH fitting (pure NumPy + optional Numba)
2. `_numba_kernels.py`: JIT-compiled Mr.ASH loop
3. `_unmappable.py`: Integration of SuSiE-inf and SuSiE-ash
4. Precision matrix computation with eigendecomposition

### Phase 5: Polish

**Priority: LOW — Quality of life.**

1. `_plotting.py`: PIP plots, diagnostic plots, changepoint plots
2. Trend filtering support
3. Comprehensive documentation (NumPy docstring format)
4. Performance benchmarks against R package

---

## 9. Testing Strategy

### 9.1 Unit Tests

Each module gets its own test file mirroring the module structure:

```
tests/
    test_ser.py              # SER correctness
    test_ibss.py             # IBSS convergence properties
    test_elbo.py             # ELBO computation
    test_credible_sets.py    # CS extraction and purity
    test_preprocessing.py    # Input validation and transformation
    test_summary_stats.py    # RSS and SS conversion
    test_mrash.py            # Mr.ASH fitting
    test_susie.py            # Integration tests for SuSiE class
    test_numerical.py        # Numerical stability edge cases
```

### 9.2 Key Properties to Test

1. **ELBO monotonicity**: ELBO should never decrease between iterations
   (tolerance: 1e-6 for floating point)
2. **Probability constraints**: alpha rows sum to 1 when null_weight=0
   (tol: 1e-10). When null_weight > 0, rows sum to <= 1 (missing mass
   equals the stripped null column's posterior weight). 0 <= PIP <= 1
3. **Dimensional correctness**: alpha is (L, p), coef is (p,), etc.
4. **Convergence**: Known easy problems should converge within max_iter
5. **Equivalence**: `fit()` and `fit_from_sufficient_stats()` produce
   identical results on the same data
6. **Null model**: When no signal, all V should be estimated near 0
7. **Known signal recovery**: On simulated data with known causal variables,
   those variables should appear in credible sets

### 9.3 Numerical Tolerances

| Quantity | Tolerance | Context |
|----------|-----------|---------|
| alpha row sums (null_weight=0) | 1e-10 | Must be very close to 1 |
| alpha row sums (null_weight>0) | 1e-10 | Must be very close to 1 minus null posterior weight |
| ELBO decrease | 1e-6 | Warning threshold |
| PIP bounds | 1e-10 | Must be in [0, 1] |
| Coefficient match (vs R) | 1e-4 | Cross-implementation |
| Convergence | 1e-3 | Default tol |
| Prior variance zero | 1e-9 | Threshold for "no effect" |

---

## 10. Dependencies

### Required
- `numpy >= 1.22`
- `scipy >= 1.9`

### Optional
- `numba >= 0.57` — For Mr.ASH acceleration (SuSiE-ash support)
- `pandas >= 1.5` — For DataFrame input/output
- `matplotlib >= 3.5` — For plotting
- `scikit-learn >= 1.1` — For sklearn estimator compatibility mixin

### Development
- `pytest`
- `pytest-cov`
- `rpy2` — For cross-validation against R package

---

## 11. Key Differences from R Package

| Aspect | susieR (R) | pysusie (Python) |
|--------|-----------|-----------------|
| API style | Functional (`susie(X, y)`) | Class-based (`SuSiE().fit(X, y)`) |
| Results | S3 list with `$` access | Dataclass with properties |
| Naming | `camelCase` / `dot.case` | `snake_case` throughout |
| Dispatch | S3 methods by data class | Single class, multiple `fit_*` methods |
| C++ code | Rcpp/RcppArmadillo | Numba JIT (optional) |
| Sparse | Matrix package | scipy.sparse |
| Plotting | Base R / ggplot2 | matplotlib |
| Defaults | `estimate_residual_variance=TRUE` for susie, `FALSE` for susie_rss | Same: True on SuSiE class, False on `fit_from_summary_stats()` parameter |

---

## 12. Example Usage

```python
import numpy as np
from pysusie import SuSiE

# --- Individual-level data ---
X = np.random.randn(500, 1000)
beta_true = np.zeros(1000)
beta_true[[100, 300, 700]] = [0.5, -0.3, 0.4]
y = X @ beta_true + np.random.randn(500)

model = SuSiE(n_effects=10, verbose=True)
model.fit(X, y)
result = model.result_

print(f"Converged in {result.n_iter} iterations")
print(f"PIPs for true causal variables: {result.pip[[100, 300, 700]]}")

cs = result.get_credible_sets(X=X)
for i, c in enumerate(cs):
    print(f"CS {i}: {c.variables}, purity={c.purity.min_abs_corr:.3f}")

# sklearn-style access also works
print(f"Coefficients: {model.coef_[:5]}")

# Predict on new data
X_new = np.random.randn(50, 1000)
y_pred = model.predict(X_new)

# --- Summary statistics ---
from pysusie import univariate_regression

stats = univariate_regression(X, y)
R = np.corrcoef(X.T)

model_rss = SuSiE(n_effects=10)
model_rss.fit_from_summary_stats(z=stats['z_scores'], R=R, n=500)
result_rss = model_rss.result_

# --- Sufficient statistics ---
# Note: center X and y first; provide means for intercept recovery.
X_centered = X - X.mean(axis=0)
y_centered = y - y.mean()
model.fit_from_sufficient_stats(
    XtX=X_centered.T @ X_centered,
    Xty=X_centered.T @ y_centered,
    yty=float(y_centered @ y_centered),
    n=500,
    X_col_means=X.mean(axis=0),
    y_mean=float(y.mean()),
)
result_ss = model.result_
```
