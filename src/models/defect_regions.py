"""Differentiable defect-region sampling for supervised class-aware alignment.

Boxes use normalized xyxy coordinates relative to the padded model input.
This samples a regular grid within each box; it is not torchvision ROIAlign.
"""
import torch
from torch import nn
from torch.nn import functional as F


class IdentityResidualAdapter(nn.Module):
    """Trainable 1x1 residual adapter, initialized to preserve source predictions."""
    def __init__(self, channels, bottleneck=32):
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.up = nn.Conv2d(bottleneck, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        return features + self.up(F.silu(self.down(features)))


def pool_defect_regions(features, boxes, batch_indices, grid_size=3):
    """Average samples within labelled boxes, then unit-normalize each vector.

    Returns (number_of_boxes, channels). Only labelled defects enter alignment;
    background and missing classes contribute no vectors. Call on each selected
    feature-map scale independently. Geometry must already include augmentation
    and letterboxing, as supplied by the detector's training batches.
    """
    if features.ndim != 4 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError('Expected BCHW feature maps and Nx4 normalized xyxy boxes')
    if grid_size < 1 or batch_indices.ndim != 1 or len(batch_indices) != len(boxes):
        raise ValueError('Invalid grid size or batch index shape')
    if not torch.isfinite(boxes).all() or ((boxes < 0) | (boxes > 1)).any():
        raise ValueError('Boxes must be finite and within [0,1]')
    if (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError('Boxes must have positive area')
    if batch_indices.dtype != torch.long:
        raise ValueError('Batch indices must be int64')
    if ((batch_indices < 0) | (batch_indices >= len(features))).any():
        raise ValueError('Batch index outside feature batch')
    if not len(boxes):
        return features.new_empty((0, features.shape[1]))
    boxes = boxes.to(device=features.device, dtype=features.dtype)
    indices = batch_indices.to(features.device)
    positions = (torch.arange(grid_size, device=features.device, dtype=features.dtype) + .5) / grid_size
    x = boxes[:, 0, None] + positions * (boxes[:, 2] - boxes[:, 0])[:, None]
    y = boxes[:, 1, None] + positions * (boxes[:, 3] - boxes[:, 1])[:, None]
    gx = x[:, None, :].expand(-1, grid_size, -1)
    gy = y[:, :, None].expand(-1, -1, grid_size)
    grid = torch.stack((gx, gy), dim=-1) * 2 - 1
    samples = F.grid_sample(features[indices], grid, mode='bilinear', padding_mode='border', align_corners=False)
    return F.normalize(samples.mean(dim=(2, 3)), dim=1, eps=1e-8)
