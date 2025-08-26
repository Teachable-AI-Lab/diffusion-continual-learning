"""
Elastic Weight Consolidation (EWC) implementation for continual learning.
"""

import torch
import copy


class EWC:
    def __init__(self, model, fisher_type, *, mu=None, c=None, diag=None):
        self.model = model                      # current, mutating model
        self.fisher_type = fisher_type          # "diag" | "rank1" | "rank1_opt"
        self.mu = mu
        self.c = c
        self.diag = diag

        # freeze a Task-0 copy and cache its flattened params
        self.model0 = copy.deepcopy(model).eval()
        self.theta0 = self._flat_unet_params(self.model0).detach()
        for p in self.model0.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _flat_unet_params(m):
        return torch.cat([p.view(-1) for p in m.unet.parameters() if p.requires_grad])

    def loss(self):
        theta = self._flat_unet_params(self.model)
        delta = theta - self.theta0.to(theta)

        if self.fisher_type == "diag":
            return 0.5 * (self.diag.to(theta) * (delta * delta)).sum()

        proj = (self.mu.to(theta) * delta).sum()
        if self.fisher_type == "rank1_opt":
            return 0.5 * float(self.c) * (proj * proj)
        else:  # "rank1"
            return 0.5 * (proj * proj)
