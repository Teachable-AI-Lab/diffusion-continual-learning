"""
Fast parameter scoring and Fisher/eigenspectrum utilities using per-sample gradients
computed in parallel with torch.func/functorch vmap.

These functions keep UNet stateless via functional_call (or functorch's make_functional)
and use vmap+grad to obtain a (B, D) matrix of per-sample parameter gradients without
looping over samples. Memory remains bounded by batch size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Hard-require torch.func (PyTorch >= 2.0)
from torch.func import vmap, grad, functional_call


@torch.no_grad()
def _maybe_to(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device) if x.device != device else x


def _build_stateless_handles(unet: torch.nn.Module):
    """Return a tuple describing how to do stateless calls and how to flatten grads.

    Returns a dict with keys:
      - mode: 'torch_func' or 'functorch'
      - params, buffers: parameter and buffer containers
      - param_names: list of parameter names (only for torch_func mode)
      - fmodel: functionalized model (only for functorch mode)
      - flatten_batch_grads: callable to flatten a batched gradient-container to (B, D)
      - call_fn: callable to execute unet with given (params, buffers, x, t, label)
    """
    # Mapping-based stateless call; preserve param order via named_parameters
    params = {n: p.detach().requires_grad_(p.requires_grad)
              for n, p in unet.named_parameters() if p.requires_grad}
    buffers = {n: b for n, b in unet.named_buffers()}
    param_names = [n for n in params.keys()]

    def call_fn(params_, buffers_, x, t, label):
        # Merge params and buffers into a single mapping as required by torch.func.functional_call
        state = {}
        state.update({n: b for n, b in buffers_.items()})
        state.update({n: p for n, p in params_.items()})
        # Try conditional call first, else unconditional
        try:
            out = functional_call(unet, state, (x, t, label), {})
        except TypeError:
            out = functional_call(unet, state, (x, t), {})
        return out.sample if hasattr(out, "sample") else out

    def flatten_batch_grads(grad_container_batched: dict[str, torch.Tensor]) -> torch.Tensor:
        # Each entry is (B, ...) -> reshape to (B, -1) and concat by param_names order
        parts = []
        for n in param_names:
            g = grad_container_batched[n]
            parts.append(g.reshape(g.shape[0], -1))
        return torch.cat(parts, dim=1)

    return {
        "params": params,
        "buffers": buffers,
        "param_names": param_names,
        "flatten_batch_grads": flatten_batch_grads,
        "call_fn": call_fn,
    }


def _make_loss_fn(call_fn):
    """Create a single-sample scalar loss fn: (params, buffers, noisy_x, t, label, target_noise) -> scalar.
    We assume noisy_x already contains the scheduler noise for timestep t.
    """

    def loss_single(params, buffers, noisy_x, t, label, target_noise):
        # Inputs arrive per-sample: noisy_x: (C,H,W); t: scalar long; label: scalar; target_noise: (C,H,W)
        # Add batch dim for UNet forward
        bx = noisy_x.unsqueeze(0)
        bt = t.unsqueeze(0)
        bl = label.unsqueeze(0)
        pred = call_fn(params, buffers, bx, bt, bl)
        # pred has shape (1,C,H,W)
        return F.mse_loss(pred, target_noise.unsqueeze(0))

    return loss_single


def _per_sample_grads_batch(unet_handles, images, labels, scheduler, time_level, device, dtype=None):
    """Compute per-sample grads for a batch via vmap+grad. Returns (B,D) tensor on device.

    Steps:
      1) Sample t per-sample (or use fixed time_level)
      2) Sample per-sample noise, construct noisy_x in batch
      3) vmap over samples of grad(loss w.r.t. params))
      4) Flatten each sample's grad container to a row vector and stack -> (B,D)
    """
    B = images.shape[0]
    # 1) timesteps
    if time_level is None:
        t_vec = torch.randint(0, 1000, (B,), device=device, dtype=torch.long)
    else:
        t_vec = torch.full((B,), int(time_level), device=device, dtype=torch.long)

    # 2) noises and noisy inputs (do it once in batch to avoid overhead inside vmap)
    noise_vec = torch.randn_like(images, device=device)
    noisy_x = scheduler.add_noise(images, noise_vec, t_vec)

    # 3) vmap per-sample grads using functional model
    params = unet_handles["params"]
    buffers = unet_handles["buffers"]
    call_fn = unet_handles["call_fn"]
    flatten_batch_grads = unet_handles["flatten_batch_grads"]

    loss_single = _make_loss_fn(call_fn)
    grad_wrt_params = grad(loss_single)

    # Vectorize across batch
    batched_grads = vmap(lambda nx, tt, lb, nz: grad_wrt_params(params, buffers, nx, tt, lb, nz))(
        noisy_x, t_vec, labels, noise_vec
    )

    # 4) flatten mapping/tuple to a (B, D) matrix
    G = flatten_batch_grads(batched_grads)
    if dtype is not None:
        G = G.to(dtype)
    return G


def compute_param_scores_fast(
    model,
    loaders_by_class,
    device: torch.device = torch.device("cuda"),
    target_class: int = 0,
    max_samples: int | None = None,
    time_level: int | None = None,
):
    """
    Vectorized per-sample gradient computation:
      returns param_scores of shape (N, D) without per-sample Python loops.
    """
    model.eval()
    model = model.to(device)
    unet = model.unet
    scheduler = model.scheduler

    # Prepare stateless UNet and flatten helpers
    handles = _build_stateless_handles(unet)

    if target_class not in loaders_by_class:
        raise KeyError(f"class_id {target_class} not found in loaders_by_class")
    loader = loaders_by_class[target_class]

    out_chunks = []
    n_collected = 0

    for images, labels in tqdm(loader, desc="param_scores_fast"):
        images = _maybe_to(images, device)
        labels = _maybe_to(labels, device)

        G = _per_sample_grads_batch(handles, images, labels, scheduler, time_level, device)
        out_chunks.append(G)
        n_collected += G.shape[0]
        if max_samples is not None and n_collected >= max_samples:
            break

    if not out_chunks:
        raise RuntimeError("Collected zero param_scores. Check your data and forward pass.")

    param_scores = torch.cat(out_chunks, dim=0)
    if max_samples is not None:
        param_scores = param_scores[:max_samples]
    return param_scores.to(device)


def compute_rank1_coeff_and_mean_fast(
    model,
    loader,
    device: torch.device = torch.device("cuda"),
    max_samples: int | None = None,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.float64,
    time_level: int | None = None,
):
    """
    Two-pass streaming computation using batched per-sample grads G (B,D):
      Pass1: mu = E[g], F_diag = E[g ⊙ g]
      Pass2: c* = E[(mu^T g)^2] / ||mu||^4
    Returns (c_star, mu, F_diag)
    """
    model.eval()
    model = model.to(device)

    handles = _build_stateless_handles(model.unet)
    scheduler = model.scheduler

    # Pass 1
    mu_sum = None
    diag_sum = None
    N = 0

    for images, labels in tqdm(loader, desc="[pass1] mu (fast)"):
        images = _maybe_to(images, device)
        labels = _maybe_to(labels, device)

        G = _per_sample_grads_batch(handles, images, labels, scheduler, time_level, device, dtype=dtype)

        if mu_sum is None:
            D = G.shape[1]
            mu_sum = torch.zeros(D, device=device, dtype=dtype)
            diag_sum = torch.zeros(D, device=device, dtype=dtype)

        mu_sum += G.sum(dim=0)
        diag_sum += (G * G).sum(dim=0)
        N += G.shape[0]

        if max_samples is not None and N >= max_samples:
            break

    if mu_sum is None or N == 0:
        raise RuntimeError("Collected zero gradients. Check data/forward pass.")

    mu = mu_sum / float(N)
    F_diag = diag_sum / float(N)
    mu_norm2 = (mu @ mu)

    if mu_norm2 <= eps:
        return torch.zeros((), device=device, dtype=dtype), mu, F_diag

    # Pass 2
    sum_proj2 = torch.zeros((), device=device, dtype=dtype)
    M = 0

    for images, labels in tqdm(loader, desc="[pass2] c* (fast)"):
        images = _maybe_to(images, device)
        labels = _maybe_to(labels, device)
        G = _per_sample_grads_batch(handles, images, labels, scheduler, time_level, device, dtype=dtype)

        proj = G @ mu  # (B,)
        sum_proj2 += (proj * proj).sum()
        M += G.shape[0]

        if max_samples is not None and M >= max_samples:
            break

    a = sum_proj2 / float(max(M, 1))
    c_star = a / (mu_norm2 * mu_norm2 + eps)
    return c_star, mu, F_diag


def compute_top_eigenpair_two_pass_fast(
    model,
    loader,
    device: torch.device = torch.device("cuda"),
    max_samples: int | None = None,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.float64,
    power_iters: int = 1,
    time_level: int | None = None,
):
    """
    Two-pass streaming power iteration for F = E[g g^T] using batched per-sample grads.
    Returns (lambda_max, u_max) where u_max is length-D eigenvector.
    """
    model.eval()
    model = model.to(device)

    handles = _build_stateless_handles(model.unet)
    scheduler = model.scheduler

    # Dimension D from params
    D = sum(p.numel() for p in handles["params"].values())

    # Pass 1: stochastic power iteration
    u = torch.randn(D, device=device, dtype=dtype)
    u = u / (torch.norm(u) + 1e-12)

    for _ in range(power_iters):
        acc = torch.zeros_like(u)
        N = 0
        for images, labels in tqdm(loader, desc="[pass1] power iter (fast)"):
            images = _maybe_to(images, device)
            labels = _maybe_to(labels, device)
            G = _per_sample_grads_batch(handles, images, labels, scheduler, time_level, device, dtype=dtype)

            acc += G.t().matmul(G.matmul(u))  # sum_i g_i (g_i^T u)
            N += G.shape[0]
            if max_samples is not None and N >= max_samples:
                break
        if N == 0:
            break
        u = acc / float(N)
        norm = torch.norm(u)
        if norm < eps:
            break
        u = u / norm

    u_max = u.clone()

    # Pass 2: Rayleigh quotient along u_max
    sum_proj2 = torch.zeros((), device=device, dtype=dtype)
    M = 0
    for images, labels in tqdm(loader, desc="[pass2] Rayleigh (fast)"):
        images = _maybe_to(images, device)
        labels = _maybe_to(labels, device)
        G = _per_sample_grads_batch(handles, images, labels, scheduler, time_level, device, dtype=dtype)

        projs = G.matmul(u_max)
        sum_proj2 += (projs * projs).sum()
        M += G.shape[0]
        if max_samples is not None and M >= max_samples:
            break

    lambda_max = (sum_proj2 / float(max(M, 1))).item()
    return lambda_max, u_max
