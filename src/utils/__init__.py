"""
Utilities package for diffusion continual learning.

This package provides organized utilities for:
- Data loading and preprocessing
- Training and evaluation
- Fisher matrix analysis
- Mathematical computations
"""

# Import key functions for backward compatibility
from .data_loading import get_cl_dataset
from .training import train_one_task, train_continual_learning
from .evaluation import FIDEvaluator
from .fisher_utils import compare_fisher_errors_streaming, plot_fisher_errors, analyze_fisher_approximations
from .ddim import create_diffusion_model, create_models_with_optimizers

__all__ = [
    'get_cl_dataset',
    'train_one_task', 
    'train_continual_learning',
    'FIDEvaluator',
    'compare_fisher_errors_streaming',
    'plot_fisher_errors',
    'analyze_fisher_approximations',
    'create_diffusion_model',
    'create_models_with_optimizers'
]
