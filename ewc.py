# ewc.py
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

class EWC:
    """
    Elastic Weight Consolidation (EWC), plain FP32.
    - Online mode keeps a single running Fisher (constant memory).
    - Classic mode stores (means, fisher) per task.

    Args:
      model: nn.Module under continual training.
      lambda_: penalty strength.
      online: True = Online EWC, False = classic multi-task EWC.
      gamma: decay for Online EWC.
      device: where to store fisher/means (CPU fine).
      ignore_keys: substrings of parameter names to skip.
    """
    def __init__(
        self,
        model: nn.Module,
        lambda_: float = 1e3,
        online: bool = True,
        gamma: float = 0.95,
        device: Optional[torch.device] = None,
        ignore_keys: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.lambda_ = float(lambda_)
        self.online = bool(online)
        self.gamma = float(gamma)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.ignore_keys = set(ignore_keys or [])

        self._targets: Dict[str, nn.Parameter] = {
            n: p for n, p in model.named_parameters()
            if p.requires_grad and not any(k in n for k in self.ignore_keys)
        }
        self._means: Dict[str, torch.Tensor] = {}
        self._fisher: Dict[str, torch.Tensor] = {}
        self._task_history: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = []

    @torch.no_grad()
    def _snapshot_means(self) -> Dict[str, torch.Tensor]:
        return {n: p.detach().clone().to(self.device) for n, p in self._targets.items()}

    def consolidate(self, dataloader, num_batches: Optional[int] = 128) -> None:
        """
        Estimate diagonal empirical Fisher on the *current* task.
        Call after finishing training that task.
        """
        dev = next(self.model.parameters()).device
        fisher_acc = {n: torch.zeros_like(p, device=dev) for n, p in self._targets.items()}
        n_seen = 0

        was_train = self.model.training
        self.model.train()

        for i, (images, labels) in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break
            images, labels = images.to(dev), labels.to(dev)

            self.model.zero_grad(set_to_none=True)
            loss = self.model.diffusion_loss(images, labels)
            loss.backward()

            for n, p in self._targets.items():
                if p.grad is not None:
                    fisher_acc[n] += p.grad.detach() ** 2
            n_seen += 1

        if n_seen == 0:
            if was_train: self.model.train()
            return

        for n in fisher_acc:
            fisher_acc[n] /= float(n_seen)

        fisher_new = {n: t.detach().to(self.device) for n, t in fisher_acc.items()}
        means_new = self._snapshot_means()

        if self.online:
            if self._fisher:
                for n in self._targets:
                    self._fisher[n] = self.gamma * self._fisher[n] + fisher_new[n]
                    self._means[n]  = means_new[n]
            else:
                self._fisher = fisher_new
                self._means  = means_new
        else:
            self._task_history.append((means_new, fisher_new))

        if was_train:
            self.model.train()

    def penalty(self) -> torch.Tensor:
        """Return λ/2 * Σ_i F_i (θ_i − θ*_i)^2 as a scalar tensor on model's device."""
        dev = next(self.model.parameters()).device
        if self.online:
            if not self._fisher:
                return torch.zeros((), device=dev)
            return self._penalty_from(self._means, self._fisher, dev)
        else:
            if not self._task_history:
                return torch.zeros((), device=dev)
            total = torch.zeros((), device=dev)
            for means_t, fisher_t in self._task_history:
                total = total + self._penalty_from(means_t, fisher_t, dev)
            return total

    def _penalty_from(self, means: Dict[str, torch.Tensor],
                      fisher: Dict[str, torch.Tensor],
                      dev: torch.device) -> torch.Tensor:
        loss = torch.zeros((), device=dev)
        for n, p in self._targets.items():
            f = fisher[n].to(dev)
            m = means[n].to(dev)
            loss = loss + (f * (p - m) ** 2).sum()
        return 0.5 * self.lambda_ * loss