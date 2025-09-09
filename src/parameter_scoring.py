"""
Parameter scoring functions for computing gradients and Fisher information.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def _maybe_to(x, device):
    return x.to(device) if x.device != device else x


def compute_param_scores(
    model,
    # t_level: int,
    loaders_by_class,          # e.g., your cl_mnist_train_loaders dict
    device: torch.device = torch.device("cuda"),
    target_class: int = 0,     # match your `if class_id != 0: continue`
    max_samples: int | None = None,
    time_level: int = None,  # if None, use random t in [0, 1000)
):
    """
    Compute per-sample parameter scores (gradients) for a DDPM-style model.

    Args:
        model: object with attributes `unet` and `scheduler`. The UNet forward should be
               compatible with (noisy_x, t[, labels]) and may return either a tensor or
               an object with `.sample`.
        t_level (int): fixed diffusion timestep to use.
        loaders_by_class (dict[int, DataLoader]): mapping class_id -> DataLoader yielding (images, labels).
        device (torch.device): device for computation.
        target_class (int): which class_id to use from `loaders_by_class`.
        max_samples (int|None): stop after collecting this many samples (None = all).

    Returns:
        param_scores: Tensor of shape (N, D) where N is number of samples processed and
                      D is the number of trainable parameters in `model.unet`.
    """
    model.eval()
    model = model.to(device)

    # Convenience handles
    unet = model.unet
    scheduler = model.scheduler

    param_scores = []
    n_collected = 0

    # Get the loader for the desired class
    if target_class not in loaders_by_class:
        raise KeyError(f"class_id {target_class} not found in loaders_by_class")
    loader = loaders_by_class[target_class]

    # Ensure autograd is enabled (we need grads!)
    # torch.set_grad_enabled(True)

    for images, labels in tqdm(loader, desc=f"param_scores"):
        images = _maybe_to(images, device)
        labels = _maybe_to(labels, device)

        # Loop per-sample to get per-sample grads (simple and faithful to your code)
        for img, label in zip(images, labels):
            img = img.unsqueeze(0)     # (1, C, H, W)
            label = label.unsqueeze(0) # (1,)

            # Fix timestep t
            # random t in [0, 1000)
            if time_level is None:
                t = torch.randint(0, 1000, (1,), device=device, dtype=torch.long)
            else:
                t = torch.tensor(time_level, device=device, dtype=torch.long)

            # Add noise
            noise = torch.randn_like(img, device=device)
            noisy_x = scheduler.add_noise(img, noise, t)

            # Forward UNet (handle conditional/unconditional + output types)
            try:
                out = unet(noisy_x, t, label)
            except TypeError:
                out = unet(noisy_x, t)

            pred_noise = out.sample if hasattr(out, "sample") else out

            # Compute per-sample loss and backward for grads w.r.t. UNet params
            unet.zero_grad()
            loss = F.mse_loss(pred_noise, noise)
            if loss.isnan() or loss.isinf():
                raise ValueError("Loss is NaN or Inf. Check your data and forward pass.")
            loss.backward()

            # Flatten/concat gradients into one vector
            grads = []
            for p in unet.parameters():
                if p.requires_grad and p.grad is not None:
                    grads.append(p.grad.reshape(-1))
            if not grads:
                raise RuntimeError("No gradients found on UNet parameters.")
            g_vec = torch.cat(grads).detach().cpu()  # (D,)
            param_scores.append(g_vec)

            n_collected += 1
            if max_samples is not None and n_collected >= max_samples:
                break

        if max_samples is not None and n_collected >= max_samples:
            break

    # Stack into (N, D)
    if len(param_scores) == 0:
        raise RuntimeError("Collected zero param_scores. Check your data and forward pass.")
    param_scores = torch.stack(param_scores, dim=0).to(device)
    if torch.any(param_scores.isnan()) or torch.any(param_scores.isinf()):
        raise ValueError("param_scores contain NaN or Inf values.")
    return param_scores


def compute_rank1_coeff_and_mean(
    model,
    # t_level: int,
    loader,                 # dict[int, DataLoader] yielding (images, labels)
    device: torch.device = torch.device("cuda"),
    # target_class: int = 0,
    max_samples: int | None = None,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.float64,  # use float64 for stable inner products
):
    """
    Streaming, two-pass computation of:
      - mu: E[g]  (flattened over all trainable UNet parameters)
      - c*: optimal rank-1 coefficient along mu mu^T:
            c* = E[(mu^T g)^2] / ||mu||^4
    No (B,D) allocation; only a running mean and scalar accumulators.

    Returns:
      c_star (scalar tensor on `device` with `dtype`)
      mu     (1-D tensor of length D on `device` with `dtype`)
    """
    model.eval()
    model = model.to(device)

    unet = model.unet
    scheduler = model.scheduler

    # if target_class not in loaders_by_class:
        # raise KeyError(f"class_id {target_class} not found in loaders_by_class")
    # loader = loaders_by_class[target_class]
    # loader = loaders_by_class

    torch.set_grad_enabled(True)

    # ---------------- PASS 1: compute mu = E[g] ----------------
    mu = None
    diag_sum = None
    N = 0

    pbar = tqdm(loader, desc=f"[pass1] mu")
    for images, labels in pbar:
        images = _maybe_to(images, device)
        labels = _maybe_to(labels, device)

        for img, label in zip(images, labels):
            img = img.unsqueeze(0)
            label = label.unsqueeze(0)

            # t = torch.full((1,), int(t_level), device=device, dtype=torch.long)
            # random t in [0, 1000)
            t = torch.randint(0, 1000, (1,), device=device, dtype=torch.long)
            noise = torch.randn_like(img, device=device)
            noisy_x = scheduler.add_noise(img, noise, t)

            try:
                out = unet(noisy_x, t, label)
            except TypeError:
                out = unet(noisy_x, t)

            pred_noise = out.sample if hasattr(out, "sample") else out

            unet.zero_grad(set_to_none=True)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            # flatten current sample's gradient
            grads = [p.grad.reshape(-1) for p in unet.parameters() if p.requires_grad and p.grad is not None]
            if not grads:
                raise RuntimeError("No gradients found on UNet parameters.")
            g = torch.cat(grads).to(dtype)

            if mu is None:
                mu = torch.zeros_like(g, device=device, dtype=dtype)
                diag_sum = torch.zeros_like(g, device=device, dtype=dtype)

            mu += g
            diag_sum += g * g  # accumulate ||g||^2 for each sample
            N += 1

            unet.zero_grad(set_to_none=True)  # free grads ASAP

            if max_samples is not None and N >= max_samples:
                break
        if max_samples is not None and N >= max_samples:
            break

    if mu is None or N == 0:
        raise RuntimeError("Collected zero gradients. Check data/forward pass.")
    mu /= float(N)
    F_diag = diag_sum / float(N)  # E[||g||^2] per parameter
    mu_norm2 = (mu @ mu)

    if mu_norm2 <= eps:
        # Degenerate direction; best rank-1 coeff is 0
        return torch.zeros((), device=device, dtype=dtype), mu

    # ---------------- PASS 2: compute a = E[(mu^T g)^2] ----------------
    sum_proj2 = torch.zeros((), device=device, dtype=dtype)
    M = 0

    pbar = tqdm(loader, desc=f"[pass2] c*")
    for images, labels in pbar:
        images = _maybe_to(images, device)
        labels = _maybe_to(labels, device)

        for img, label in zip(images, labels):
            img = img.unsqueeze(0)
            label = label.unsqueeze(0)

            # t = torch.full((1,), int(t_level), device=device, dtype=torch.long)
            # random t in [0, 1000)
            t = torch.randint(0, 1000, (1,), device=device, dtype=torch.long)
            noise = torch.randn_like(img, device=device)
            noisy_x = scheduler.add_noise(img, noise, t)

            try:
                out = unet(noisy_x, t, label)
            except TypeError:
                out = unet(noisy_x, t)
            pred_noise = out.sample if hasattr(out, "sample") else out

            unet.zero_grad(set_to_none=True)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            grads = [p.grad.reshape(-1) for p in unet.parameters() if p.requires_grad and p.grad is not None]
            g = torch.cat(grads).to(dtype)

            s = (g @ mu)            # scalar projection
            sum_proj2 += s * s      # accumulate (mu^T g)^2
            M += 1

            unet.zero_grad(set_to_none=True)

            if max_samples is not None and M >= max_samples:
                break
        if max_samples is not None and M >= max_samples:
            break

    a = sum_proj2 / float(M)        # a = E[(mu^T g)^2]
    c_star = a / (mu_norm2 * mu_norm2 + eps)

    return c_star, mu, F_diag


def compute_top_eigenpair_two_pass(
    model,
    loader,
    device: torch.device = torch.device("cuda"),
    max_samples: int | None = None,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.float64,
    power_iters: int = 1,  # number of stochastic power iterations in pass 1
):
    """
    Two-pass streaming computation of the top eigenvector and eigenvalue of
    F = E[g g^T] for UNet gradients g.
    
    Pass 1: stochastic power iteration to estimate u_max
    Pass 2: compute Rayleigh quotient along u_max to estimate lambda_max

    Returns:
        u_max       : top eigenvector of F (length D)
        lambda_max  : top eigenvalue of F
    """
    model.eval()
    model = model.to(device)

    # ---------------- PASS 1: stochastic power iteration ----------------
    D = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
    u = torch.randn(D, device=device, dtype=dtype)
    u /= torch.norm(u)

    for _ in range(power_iters):
        N = 0
        acc = torch.zeros_like(u, device=device, dtype=dtype)

        for images, labels in loader:
            images = _maybe_to(images, device)
            labels = _maybe_to(labels, device)
            for img, label in zip(images, labels):
                img = img.unsqueeze(0)
                label = label.unsqueeze(0)

                t = torch.randint(0, 1000, (1,), device=device, dtype=torch.long)
                noise = torch.randn_like(img, device=device)
                noisy_x = model.scheduler.add_noise(img, noise, t)

                try:
                    out = model.unet(noisy_x, t, label)
                except TypeError:
                    out = model.unet(noisy_x, t)
                pred_noise = out.sample if hasattr(out, "sample") else out

                model.unet.zero_grad(set_to_none=True)
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()

                grads = [p.grad.reshape(-1) for p in model.unet.parameters()
                         if p.requires_grad and p.grad is not None]
                g = torch.cat(grads).to(dtype)

                # stochastic power iteration update
                acc += g * (g @ u)
                N += 1

                model.unet.zero_grad(set_to_none=True)

                if max_samples is not None and N >= max_samples:
                    break
            if max_samples is not None and N >= max_samples:
                break

        # normalize after the pass
        if N > 0:
            u = acc / N
            norm = torch.norm(u)
            if norm < eps:
                break
            u /= norm

    u_max = u.clone()

    # ---------------- PASS 2: compute Rayleigh quotient ----------------
    sum_proj2 = torch.zeros((), device=device, dtype=dtype)
    M = 0

    for images, labels in loader:
        images = _maybe_to(images, device)
        labels = _maybe_to(labels, device)
        for img, label in zip(images, labels):
            img = img.unsqueeze(0)
            label = label.unsqueeze(0)

            t = torch.randint(0, 1000, (1,), device=device, dtype=torch.long)
            noise = torch.randn_like(img, device=device)
            noisy_x = model.scheduler.add_noise(img, noise, t)

            try:
                out = model.unet(noisy_x, t, label)
            except TypeError:
                out = model.unet(noisy_x, t)
            pred_noise = out.sample if hasattr(out, "sample") else out

            model.unet.zero_grad(set_to_none=True)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            grads = [p.grad.reshape(-1) for p in model.unet.parameters()
                     if p.requires_grad and p.grad is not None]
            g = torch.cat(grads).to(dtype)

            # accumulate (u_max^T g)^2
            s = g @ u_max
            sum_proj2 += s * s
            M += 1

            model.unet.zero_grad(set_to_none=True)

            if max_samples is not None and M >= max_samples:
                break
        if max_samples is not None and M >= max_samples:
            break

    lambda_max = (sum_proj2 / max(M, 1)).item()

    return lambda_max, u_max


# def compute_top_eigenpair_two_pass(
#     model,
#     loader,
#     device: torch.device = torch.device("cuda"),
#     max_samples: int | None = None,
#     eps: float = 1e-12,
#     dtype: torch.dtype = torch.float64,
#     power_iters: int = 1,   # number of passes of power iteration before final combined pass
# ):
#     """
#     Multi-pass streaming computation of the top eigenvector and eigenvalue of
#     F = E[g g^T] for UNet gradients g.

#     Passes 1..n: stochastic power iteration to refine u
#     Final pass:  do one more power iteration update and accumulate Rayleigh quotient

#     Returns:
#         u_max       : top eigenvector of F (length D)
#         lambda_max  : top eigenvalue of F
#     """
#     model.eval()
#     model = model.to(device)

#     # ---------------- INIT ----------------
#     D = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
#     u = torch.randn(D, device=device, dtype=dtype)
#     u /= torch.norm(u)

#     # ---------------- PASSES 1..n: Power iteration ----------------
#     for p_iter in range(power_iters):
#         acc = torch.zeros_like(u, device=device, dtype=dtype)
#         N = 0
#         pbar = tqdm(loader, desc=f"[pass {p_iter+1}/{power_iters}] power iter")
#         for images, labels in pbar:
#             images = _maybe_to(images, device)
#             labels = _maybe_to(labels, device)
#             for img, label in zip(images, labels):
#                 img = img.unsqueeze(0)
#                 label = label.unsqueeze(0)

#                 t = torch.randint(0, 1000, (1,), device=device, dtype=torch.long)
#                 noise = torch.randn_like(img, device=device)
#                 noisy_x = model.scheduler.add_noise(img, noise, t)

#                 try:
#                     out = model.unet(noisy_x, t, label)
#                 except TypeError:
#                     out = model.unet(noisy_x, t)
#                 pred_noise = out.sample if hasattr(out, "sample") else out

#                 model.unet.zero_grad(set_to_none=True)
#                 loss = F.mse_loss(pred_noise, noise)
#                 loss.backward()

#                 grads = [p.grad.reshape(-1) for p in model.unet.parameters()
#                          if p.requires_grad and p.grad is not None]
#                 g = torch.cat(grads).to(dtype)

#                 acc += g * (g @ u)
#                 N += 1

#                 model.unet.zero_grad(set_to_none=True)

#                 if max_samples is not None and N >= max_samples:
#                     break
#             if max_samples is not None and N >= max_samples:
#                 break

#         if N > 0:
#             u = acc / N
#             norm = torch.norm(u)
#             if norm < eps:
#                 break
#             u /= norm

#     # ---------------- FINAL PASS: Power + Rayleigh ----------------
#     acc = torch.zeros_like(u, device=device, dtype=dtype)
#     sum_proj2 = torch.zeros((), device=device, dtype=dtype)
#     M = 0
#     pbar = tqdm(loader, desc="[final pass] power + Rayleigh")
#     for images, labels in pbar:
#         images = _maybe_to(images, device)
#         labels = _maybe_to(labels, device)
#         for img, label in zip(images, labels):
#             img = img.unsqueeze(0)
#             label = label.unsqueeze(0)

#             t = torch.randint(0, 1000, (1,), device=device, dtype=torch.long)
#             noise = torch.randn_like(img, device=device)
#             noisy_x = model.scheduler.add_noise(img, noise, t)

#             try:
#                 out = model.unet(noisy_x, t, label)
#             except TypeError:
#                 out = model.unet(noisy_x, t)
#             pred_noise = out.sample if hasattr(out, "sample") else out

#             model.unet.zero_grad(set_to_none=True)
#             loss = F.mse_loss(pred_noise, noise)
#             loss.backward()

#             grads = [p.grad.reshape(-1) for p in model.unet.parameters()
#                      if p.requires_grad and p.grad is not None]
#             g = torch.cat(grads).to(dtype)

#             # Power update contribution
#             acc += g * (g @ u)

#             # Rayleigh quotient contribution
#             s = g @ u
#             sum_proj2 += s * s
#             M += 1

#             model.unet.zero_grad(set_to_none=True)

#             if max_samples is not None and M >= max_samples:
#                 break
#         if max_samples is not None and M >= max_samples:
#             break

#     # # Normalize eigenvector after final update
#     # if M > 0:
#     #     u = acc / M
#     #     norm = torch.norm(u)
#     #     if norm > eps:
#     #         u /= norm

#     u_max = u.clone()
#     lambda_max = (sum_proj2 / max(M, 1)).item()

#     return lambda_max, u_max