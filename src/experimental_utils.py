"""
Experimental utilities for Fisher matrix error comparison and analysis.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


@torch.no_grad()
def _maybe_to(x, device):
    return x.to(device) if x.device != device else x


def compare_fisher_errors_streaming(
    model,
    t_level: int,
    loaders_by_class,                  # dict[int, DataLoader] -> (images, labels)
    mu: torch.Tensor,                  # v (flattened) in the same param order as unet.parameters()
    c_star: float,                     # optimal scalar coefficient
    target_class: int = 0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float64,
    max_samples: int | None = None,
    num_diag_probes: int = 15,          # z-probes for ||diag(E[g⊙g])||_F^2
):
    """
    Returns Frobenius errors (absolute and relative) for:
      - rank1 (c=1)
      - rank1_opt (c=c_star)
      - diag (estimated)
    Also returns components: vFv, ||F||_F, ||D||_F, ||v||^2, tr(F).
    """
    model.eval(); model = model.to(device)
    unet, scheduler = model.unet, model.scheduler

    # data loader
    if target_class not in loaders_by_class:
        raise KeyError(f"class_id {target_class} not found")
    loader = loaders_by_class[target_class]

    # v = mu (flattened), constants
    v = mu.detach().to(dtype).cpu()    # compute on CPU (memory-friendly)
    v_norm2 = float(v.dot(v))          # ||v||^2
    c_star = float(c_star)

    # Pre-generate Rademacher probes z for diagonal norm (CPU, int8 -> lightweight)
    D = v.numel()
    z_list = [torch.randint(0, 2, (D,), dtype=torch.int8).mul_(2).sub_(1) for _ in range(num_diag_probes)]

    # Accumulators
    N = 0
    sum_s2 = 0.0                       # for v^T F v = E[(g^T v)^2]
    sum_pair_sq = 0.0                  # for ||F||_F^2 = E[(g^T g')^2]
    num_pairs = 0
    u_tot = [0.0 for _ in range(num_diag_probes)]  # for ||diag(E[g⊙g])||_F via Hutchinson on diag
    trace_sum = 0.0                    # tr(F) = E[||g||^2]

    prev_g = None                      # hold one g to form a pair (adjacent samples)

    torch.set_grad_enabled(True)
    pbar = tqdm(loader, desc=f"compare_F_errors@t={t_level}")

    for images, labels in pbar:
        images = images.to(device); labels = labels.to(device)
        for img, lab in zip(images, labels):
            img = img.unsqueeze(0); lab = lab.unsqueeze(0)
            t = torch.full((1,), int(t_level), device=device, dtype=torch.long)

            # Draw noise, make noisy input
            noise = torch.randn_like(img, device=device)
            noisy_x = scheduler.add_noise(img, noise, t)

            # Forward UNet (conditional/unconditional)
            try:
                out = unet(noisy_x, t, lab)
            except TypeError:
                out = unet(noisy_x, t)
            pred_noise = out.sample if hasattr(out, "sample") else out

            # Per-sample MSE and grad wrt UNet params
            unet.zero_grad(set_to_none=True)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            # Flatten grad to CPU (float64) — one vector only
            g_list = [p.grad.reshape(-1) for p in unet.parameters() if p.requires_grad and p.grad is not None]
            if not g_list: raise RuntimeError("No grads on UNet parameters.")
            g = torch.cat(g_list).detach().cpu().to(dtype)

            # v^T F v = E[(g^T v)^2]
            s = float(g.dot(v))
            sum_s2 += s * s

            # tr(F) = E[||g||^2]
            trace_sum += float(g.pow(2).sum())

            # ||diag(E[g⊙g])||_F^2 via Hutchinson-on-diagonal:
            # For each probe z: m_z = E[(g⊙g)^T z] ≈ (1/N) Σ (g⊙g)^T z  ; then ||diag||_F^2 ≈ mean_k m_z^2
            g2 = g * g
            for k, z in enumerate(z_list):
                # cast z to float64 on the fly (keeps memory low)
                u_tot[k] += float((g2 * z.to(torch.float64)).sum())

            # ||F||_F^2 via adjacent independent pairs: E[(g^T g')^2]
            if prev_g is None:
                prev_g = g
            else:
                dot = float(prev_g.dot(g))
                sum_pair_sq += dot * dot
                num_pairs += 1
                prev_g = None  # disjoint pairs: (1,2), (3,4), ...

            N += 1
            unet.zero_grad(set_to_none=True)

            if max_samples is not None and N >= max_samples:
                break
        if max_samples is not None and N >= max_samples:
            break

    if N == 0:
        raise RuntimeError("No samples processed.")

    # Estimates
    vFv = sum_s2 / N
    trF = trace_sum / N

    # ||diag(E[g⊙g])||_F^2
    m_sq = [(u / N) ** 2 for u in u_tot]                 # m_z^2
    diagF_frob_sq = sum(m_sq) / max(num_diag_probes, 1)  # average over probes

    # ||F||_F^2
    if num_pairs == 0:
        raise RuntimeError("Need at least one pair to estimate ||F||_F^2. Increase max_samples.")
    frobF_sq = sum_pair_sq / num_pairs

    # Errors (Frobenius)
    err_rank1_sq      = frobF_sq - 2.0 * vFv + (v_norm2 ** 2)
    err_rank1opt_sq   = frobF_sq - 2.0 * c_star * vFv + (c_star ** 2) * (v_norm2 ** 2)
    err_diag_sq       = frobF_sq - diagF_frob_sq

    # Stabilize small negatives due to MC noise
    clip = lambda x: float(torch.clamp(torch.tensor(x, dtype=dtype), min=0.0))
    err_rank1    = clip(err_rank1_sq)    ** 0.5
    err_rank1opt = clip(err_rank1opt_sq) ** 0.5
    err_diag     = clip(err_diag_sq)     ** 0.5
    frobF        = frobF_sq ** 0.5

    return {
        "components": {
            "v_norm2": v_norm2,
            "vFv": vFv,
            "trF": trF,
            "||F||_F": frobF,
            "||diag(E[g⊙g])||_F": diagF_frob_sq ** 0.5,
            "num_samples": N,
            "num_pairs": num_pairs,
            "num_diag_probes": num_diag_probes,
        },
        "errors_frobenius": {
            "rank1_c1": {"abs": err_rank1,    "rel": err_rank1 / (frobF + 1e-12)},
            "rank1_c*": {"abs": err_rank1opt, "rel": err_rank1opt / (frobF + 1e-12), "c*": c_star},
            "diag":     {"abs": err_diag,     "rel": err_diag / (frobF + 1e-12)},
        },
    }


def plot_fisher_errors(t_levels, diag_errors, rank1_errors, rank1_optimal_errors, save_path=None):
    """
    Plot Fisher matrix approximation errors across different timestep levels.
    
    Args:
        t_levels: List of timestep levels
        diag_errors: List of diagonal approximation errors
        rank1_errors: List of rank-1 approximation errors  
        rank1_optimal_errors: List of optimal rank-1 approximation errors
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(t_levels, np.array(diag_errors)*10000, label='Diagonal Error', marker='o')
    plt.plot(t_levels, np.array(rank1_errors)*10000, label='Rank-1 Error', marker='o')
    plt.plot(t_levels, np.array(rank1_optimal_errors)*10000, label='Optimal Rank-1 Error', marker='o')
    plt.yscale('log')
    plt.xlabel('Timestep Level')
    plt.ylabel('Error Norm')
    plt.title('EWC Fisher Matrix Approximation Errors')
    plt.legend()
    plt.grid()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def analyze_fisher_approximations(param_scores, Fisher=None):
    """
    Analyze different Fisher matrix approximations.
    
    Args:
        param_scores: Parameter scores tensor (N, D)
        Fisher: Optional pre-computed Fisher matrix (D, D)
    
    Returns:
        Dictionary with analysis results
    """
    from .fisher_analysis import optimal_rank1_coeff, empirical_fisher_dense
    
    if Fisher is None:
        Fisher = empirical_fisher_dense(param_scores).to('cpu')
    
    c, mu = optimal_rank1_coeff(param_scores, eps=1e-12, use_float64=False)

    F_diag = torch.diag(torch.diag(Fisher))
    err_diag = torch.linalg.norm(Fisher - F_diag)

    # rank-1 as score
    F_r1_score = mu.unsqueeze(1) @ mu.unsqueeze(0)  # (D, D)
    err_r1_score = torch.linalg.norm(Fisher - F_r1_score)

    F_r1_optimal = mu.unsqueeze(1) @ mu.unsqueeze(0) * c  # (D, D)
    err_r1_optimal = torch.linalg.norm(Fisher - F_r1_optimal)
    
    return {
        'diagonal_error': err_diag.item(),
        'rank1_error': err_r1_score.item(),
        'rank1_optimal_error': err_r1_optimal.item(),
        'optimal_coefficient': c.item(),
        'mu': mu,
        'Fisher': Fisher,
        'F_diag': F_diag,
        'F_r1_score': F_r1_score,
        'F_r1_optimal': F_r1_optimal
    }
