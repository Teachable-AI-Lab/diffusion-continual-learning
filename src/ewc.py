"""
Elastic Weight Consolidation (EWC) – full history (sum over past tasks).
"""

import torch


class EWC:
    def __init__(self, teacher_model, fisher_type, mu=None, c=None, diag=None):
        # fisher_type: "diag" | "rank1" | "rank1_opt"
        self.fisher_type = fisher_type
        self.tasks = []
        self.add_task(teacher_model, mu=mu, c=c, diag=diag)

    @staticmethod
    def _flat_unet_params(m):
        return torch.cat([p.view(-1) for p in m.unet.parameters()])

    def add_task(self, teacher_model, *, mu=None, c=None, diag=None):
        # Cache flattened params of the frozen teacher for this task
        theta0 = self._flat_unet_params(teacher_model.eval()).detach()
        self.tasks.append((theta0, mu, c, diag))

    def loss(self, model):
        theta = self._flat_unet_params(model)
        total = 0.0
        for (theta0, mu, c, diag) in self.tasks:
            delta = theta - theta0.to(theta)
            if self.fisher_type == "diag":
                total = total + 0.5 * (diag.to(theta) * (delta * delta)).sum()
            else:
                proj = (mu.to(theta) * delta).sum()
                if self.fisher_type == "rank1_opt":
                    total = total + 0.5 * float(c) * (proj * proj)
                else:  # "rank1"
                    total = total + 0.5 * (proj * proj)
        return total
