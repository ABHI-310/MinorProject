"""Small adapters used to align source and target YOLO neck features."""

from __future__ import annotations

import torch
from torch import nn


class LightweightAdapter(nn.Module):
    """The requested FC adapter, for pooled (B, D) feature vectors."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected pooled (B, D) features, got {tuple(features.shape)}")
        return self.network(features)


class FeatureMapAdapter(nn.Module):
    """Residual 1×1 adapter for a YOLO FPN map, suitable for arbitrary map size."""

    def __init__(self, channels: int, bottleneck: int | None = None, dropout: float = 0.1):
        super().__init__()
        bottleneck = bottleneck or max(16, channels // 4)
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(bottleneck, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)


def pooled_features(feature_maps: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
    """Global-average-pool one or more FPN maps to a single feature vector."""
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = [feature_maps]
    pooled = [torch.mean(feature, dim=(2, 3)) for feature in feature_maps]
    return torch.cat(pooled, dim=1)
