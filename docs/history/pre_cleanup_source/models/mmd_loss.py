"""Class-aware, numerically stable RBF Maximum Mean Discrepancy loss."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class DefectAwareMMDLoss(nn.Module):
    def __init__(self, class_weights: Mapping[int, float] | None = None, kernel_sigma: float = 1.0):
        super().__init__()
        if kernel_sigma <= 0:
            raise ValueError("kernel_sigma must be positive")
        self.class_weights = dict(class_weights or {})
        self.kernel_sigma = float(kernel_sigma)

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(x, y, p=2).square()
        return torch.exp(-distances / (2 * self.kernel_sigma**2))

    def mmd_squared(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(source) == 0 or len(target) == 0:
            return source.new_zeros(())
        return self.rbf_kernel(source, source).mean() + self.rbf_kernel(target, target).mean() - 2 * self.rbf_kernel(source, target).mean()

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_labels: torch.Tensor | None = None,
        target_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Align all samples if labels are absent; otherwise align shared classes only."""
        if source_labels is None or target_labels is None:
            return self.mmd_squared(source, target).clamp_min(0)
        shared = sorted(set(source_labels.detach().cpu().tolist()) & set(target_labels.detach().cpu().tolist()))
        terms: list[torch.Tensor] = []
        weights: list[float] = []
        for class_id in shared:
            s = source[source_labels == class_id]
            t = target[target_labels == class_id]
            if len(s) and len(t):
                terms.append(self.mmd_squared(s, t).clamp_min(0))
                weights.append(float(self.class_weights.get(int(class_id), 1.0)))
        if not terms:
            return source.new_zeros(())
        weighted = sum(loss * weight for loss, weight in zip(terms, weights))
        return weighted / sum(weights)
