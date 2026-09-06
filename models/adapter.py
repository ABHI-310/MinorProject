"""
adapter.py
Lightweight 2-layer Fully Connected Adapter for domain adaptation in PCB defect detection.
Bridges representations between source (2-layer synthetic) and target (multi-layer manufacturing).
"""

import torch
import torch.nn as nn


class LightweightAdapter(nn.Module):
    """
    Lightweight Domain Adapter module:
    - FC1: input_dim -> hidden_dim (e.g., 2048 -> 512)
    - BatchNorm1d
    - ReLU
    - Dropout(0.2)
    - FC2: hidden_dim -> output_dim (e.g., 512 -> 128)
    - BatchNorm1d
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Global average pool if incoming tensor is 4D (B, C, H, W)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle 4D feature map from conv backbone
        if x.dim() == 4:
            x = self.gap(x)
            x = torch.flatten(x, 1)
        elif x.dim() > 2:
            x = torch.flatten(x, 1)

        # Dynamic dimension match if input dimension differs
        if x.size(1) != self.input_dim:
            if not hasattr(self, "_proj") or self._proj.in_features != x.size(1):
                self._proj = nn.Linear(x.size(1), self.input_dim).to(x.device)
            x = self._proj(x)

        x = self.fc1(x)
        # Avoid batch norm crash if batch size is 1 during inference
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)

        return x

