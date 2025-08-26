"""
Fisher matrix analysis utilities including empirical Fisher computation and rank-1 approximations.
"""

import math
import torch


@torch.no_grad()
def empirical_fisher_dense(param_scores: torch.Tensor) -> torch.Tensor:
    """
    param_scores: (B, D) on CUDA, dtype=torch.float32 (recommended)
    Returns Fisher: (D, D) on CUDA, float32
    """
    assert param_scores.is_cuda, "Move param_scores to CUDA"
    # Enable TF32 for big speedups on Ampere+ with minor precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    X = param_scores.contiguous()                            # (B, D)
    B = X.shape[0]
    # Pre-scale to reduce one kernel and improve numerical stability
    X = X * (1.0 / math.sqrt(B))
    F = X.T @ X                                              # (D, D)
    return F


def optimal_rank1_coeff(param_scores: torch.Tensor, eps: float = 1e-12, use_float64: bool = True):
    """
    param_scores: (B, D) tensor where each row is a per-sample parameter score/grad g_i
    Returns:
      c_star: scalar tensor (optimal coefficient for c * μ μ^T)
      mu:     (D,) tensor, the batch mean μ
    Computes c* = E[(μ^T g)^2] / ||μ||^4, with E taken over the batch,
    avoiding any D×D matrix construction.
    """
    x = param_scores
    if use_float64:
        x = x.to(torch.float64)

    # μ = E[g]
    mu = x.mean(dim=0)                                    # (D,)
    mu_norm2 = mu.dot(mu)                                 # ||μ||^2

    # Handle μ ≈ 0 safely (then the best scalar along μμ^T is 0)
    if mu_norm2 <= eps:
        c_star = torch.zeros((), device=x.device, dtype=x.dtype)
        return c_star.to(param_scores.dtype), mu.to(param_scores.dtype)

    # μ^T F μ = E[(μ^T g)^2] = mean of squared projections of g on μ
    proj = x @ mu                                         # (B,)
    a = (proj * proj).mean()                              # μ^T F μ

    # c* = a / ||μ||^4
    c_star = a / (mu_norm2 * mu_norm2)

    return c_star.to('cpu'), mu.to('cpu')
