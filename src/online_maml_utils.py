"""Helper utilities for first-order online MAML training.

Implements task-level inner-loop adaptation and meta-updates that follow
Finn et al. (2019), "Online Meta-Learning" (https://proceedings.mlr.press/v97/finn19a/finn19a.pdf).
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from copy import deepcopy
from typing import Iterable, Iterator, Sequence, Tuple

import torch
import torch.nn.functional as F

try:  # defer requirement unless second-order is requested
    from torch.nn.utils.stateless import functional_call
    _FUNCTIONAL_CALL_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - depends on torch version
    functional_call = None  # type: ignore
    _FUNCTIONAL_CALL_ERROR = exc

Batch = Tuple[torch.Tensor, torch.Tensor]


def clone_model_for_adaptation(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Create a fast copy of the model that can be adapted independently."""
    fast_model = deepcopy(model)
    fast_model.to(device)
    fast_model.train()
    for param in fast_model.parameters():
        param.requires_grad_(True)
    return fast_model


def _ddim_reconstruction_loss(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the MSE diffusion loss used for both inner and meta steps."""
    timesteps, noise, _, model_pred = model.diffusion_loss(images, labels)
    return F.mse_loss(model_pred, noise, reduction="mean")


def run_inner_loop(
    fast_model: torch.nn.Module,
    support_batches: Sequence[Batch],
    inner_lr: float,
    device: torch.device,
    grad_clip: float | None = None,
) -> float:
    """Adapt the fast copy on the support batches using SGD."""
    if not support_batches:
        return 0.0
    inner_opt = torch.optim.SGD(fast_model.parameters(), lr=inner_lr)
    last_loss = 0.0
    for images, labels in support_batches:
        images = images.to(device)
        labels = labels.to(device)
        inner_opt.zero_grad()
        loss = _ddim_reconstruction_loss(fast_model, images, labels)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(fast_model.parameters(), grad_clip)
        inner_opt.step()
        last_loss = float(loss.item())
    return last_loss


def meta_update(
    base_model: torch.nn.Module,
    fast_model: torch.nn.Module,
    query_batches: Sequence[Batch],
    outer_optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> float | None:
    """Backprop through the adapted copy and apply the gradients to the base model."""
    if not query_batches:
        return None
    fast_model.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_batches = 0
    for images, labels in query_batches:
        images = images.to(device)
        labels = labels.to(device)
        loss = _ddim_reconstruction_loss(fast_model, images, labels)
        loss.backward()
        total_loss += float(loss.item())
        total_batches += 1
    for param, fast_param in zip(base_model.parameters(), fast_model.parameters()):
        grad = fast_param.grad
        if grad is None:
            continue
        if param.grad is None:
            param.grad = grad.detach().clone()
        else:
            param.grad.copy_(grad.detach())
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip)
    outer_optimizer.step()
    outer_optimizer.zero_grad(set_to_none=True)
    if total_batches == 0:
        return None
    return total_loss / total_batches


def _prepare_diffusion_inputs(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = images.device
    noise = torch.randn_like(images)
    bsz = images.size(0)
    timesteps = torch.randint(
        0,
        model.scheduler.config.num_train_timesteps,
        (bsz,),
        device=device,
        dtype=torch.long,
    )
    noisy_images = model.scheduler.add_noise(images, noise, timesteps)
    class_labels = labels.to(device)
    return noise, timesteps, noisy_images, class_labels


def _stateless_unet_forward(
    model: torch.nn.Module,
    params: OrderedDict[str, torch.Tensor],
    buffers: OrderedDict[str, torch.Tensor],
    noisy_images: torch.Tensor,
    timesteps: torch.Tensor,
    class_labels: torch.Tensor,
) -> torch.Tensor:
    if functional_call is None:
        raise RuntimeError(
            "Second-order mode requires torch.nn.utils.stateless.functional_call to be available"
        ) from _FUNCTIONAL_CALL_ERROR
    state = OrderedDict(params)
    if buffers:
        for name, buf in buffers.items():
            if name not in state:
                state[name] = buf
    outputs = functional_call(model.unet, state, (noisy_images, timesteps, class_labels))
    return outputs.sample if hasattr(outputs, "sample") else outputs


def _ddim_reconstruction_loss_functional(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    params: OrderedDict[str, torch.Tensor],
    buffers: OrderedDict[str, torch.Tensor],
) -> torch.Tensor:
    noise, timesteps, noisy_images, class_labels = _prepare_diffusion_inputs(model, images, labels)
    model_pred = _stateless_unet_forward(model, params, buffers, noisy_images, timesteps, class_labels)
    return F.mse_loss(model_pred, noise, reduction="mean")


def run_functional_inner_loop(
    model: torch.nn.Module,
    support_batches: Sequence[Batch],
    inner_lr: float,
    use_second_order: bool,
) -> Tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor], float]:
    """Perform inner-loop adaptation with stateless weights for higher-order gradients."""
    if not support_batches:
        buffers = OrderedDict(model.unet.named_buffers())
        params = OrderedDict(model.unet.named_parameters())
        return params, buffers, 0.0

    buffers = OrderedDict(model.unet.named_buffers())
    fast_params = OrderedDict(model.unet.named_parameters())
    last_loss = 0.0

    for images, labels in support_batches:
        loss = _ddim_reconstruction_loss_functional(model, images, labels, fast_params, buffers)
        grads = torch.autograd.grad(
            loss,
            tuple(fast_params.values()),
            create_graph=use_second_order,
        )
        fast_params = OrderedDict(
            (name, param - inner_lr * grad)
            for (name, param), grad in zip(fast_params.items(), grads)
        )
        if not use_second_order:
            fast_params = OrderedDict((name, param.detach()) for name, param in fast_params.items())
        last_loss = float(loss.item())

    return fast_params, buffers, last_loss


def meta_update_functional(
    model: torch.nn.Module,
    fast_params: OrderedDict[str, torch.Tensor],
    buffers: OrderedDict[str, torch.Tensor],
    query_batches: Sequence[Batch],
    outer_optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
) -> float | None:
    """Compute meta-gradient via functional weights for second-order MAML."""
    if not query_batches:
        return None

    losses = []
    for images, labels in query_batches:
        loss = _ddim_reconstruction_loss_functional(model, images, labels, fast_params, buffers)
        losses.append(loss)

    if not losses:
        return None

    meta_loss = torch.stack(losses).mean()
    outer_optimizer.zero_grad(set_to_none=True)
    meta_loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    outer_optimizer.step()
    return float(meta_loss.item())


def support_query_batches(
    loader: Iterable[Batch],
    support_batches: int,
    query_batches: int,
    max_pairs: int | None = None,
) -> Iterator[Tuple[Sequence[Batch], Sequence[Batch]]]:
    """Yield (support, query) mini-batch blocks for online meta-updates."""
    if support_batches <= 0:
        raise ValueError("support_batches must be at least 1")
    if query_batches <= 0:
        raise ValueError("query_batches must be at least 1")
    iterator = iter(loader)
    emitted = 0
    while True:
        support = list(itertools.islice(iterator, support_batches))
        query = list(itertools.islice(iterator, query_batches))
        if not support or not query:
            break
        yield support, query
        emitted += 1
        if max_pairs is not None and emitted >= max_pairs:
            break
