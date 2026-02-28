"""Public package interface for pysusie."""

from ._credible_sets import compute_purity, extract_credible_sets
from ._mrash import fit_mrash
from ._preprocessing import (
    compute_sufficient_stats,
    estimate_ld_regularization,
    preprocess_individual_data,
    preprocess_summary_stats,
    preprocess_sufficient_stats,
    univariate_regression,
)
from ._types import CredibleSet, MrASHResult, PurityMetrics, SuSiEResult
from ._trendfilter import TrendFilterResult, fit_trendfilter, trend_filter_design
from ._unmappable import SuSiEAshResult, SuSiEInfResult, susie_ash, susie_inf, susie_inf_precision
from .datasets import load_example
from .susie import SuSiE, susie_auto

__all__ = [
    "SuSiE",
    "SuSiEResult",
    "CredibleSet",
    "PurityMetrics",
    "MrASHResult",
    "susie_auto",
    "compute_sufficient_stats",
    "univariate_regression",
    "estimate_ld_regularization",
    "preprocess_individual_data",
    "preprocess_sufficient_stats",
    "preprocess_summary_stats",
    "extract_credible_sets",
    "compute_purity",
    "fit_mrash",
    "TrendFilterResult",
    "trend_filter_design",
    "fit_trendfilter",
    "SuSiEInfResult",
    "SuSiEAshResult",
    "susie_inf_precision",
    "susie_inf",
    "susie_ash",
    "load_example",
]
