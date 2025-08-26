"""
Diffusion Continual Learning Package

This package contains utilities for continual learning with diffusion models,
including EWC (Elastic Weight Consolidation) implementations and Fisher
information analysis tools.
"""

from .utils.parameter_scoring import compute_param_scores, compute_rank1_coeff_and_mean
from .utils.ewc import EWC
from .utils.fisher_analysis import empirical_fisher_dense, optimal_rank1_coeff
from .utils.fisher_utils import compare_fisher_errors_streaming, plot_fisher_errors, analyze_fisher_approximations

__version__ = "0.1.0"

__all__ = [
    "compute_param_scores",
    "compute_rank1_coeff_and_mean", 
    "EWC",
    "empirical_fisher_dense",
    "optimal_rank1_coeff",
    "compare_fisher_errors_streaming",
    "plot_fisher_errors",
    "analyze_fisher_approximations"
]
