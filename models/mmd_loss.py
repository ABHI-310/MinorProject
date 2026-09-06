"""
mmd_loss.py
Defect-Aware Maximum Mean Discrepancy (MMD) Loss for domain adaptation.
Weights alignment based on defect criticality (solder bridging > missing component > trace > cosmetic).
"""

import torch
import torch.nn as nn


class DefectAwareMMLoss(nn.Module):
    """
    Defect-aware MMD loss: prioritizes alignment of defect-critical classes
    L_mmd = Σ_c (α_c * MMD²(X_s^c, X_t^c))

    where:
    - c = defect class (mapped to 6-class space)
    - α_c = class weight (higher for critical defects)
    - MMD = kernel mean embedding distance (Gaussian RBF kernel)
    """

    def __init__(self, class_weights: dict = None, kernel_sigma: float = 1.0, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.kernel_sigma = kernel_sigma

        # Default defect criticality weighting:
        # Class 3 (short/solder defect) -> 1.0 (highest critical)
        # Class 0 (missing hole / void) -> 0.95
        # Class 2 (open / broken trace) -> 0.90
        # Class 1 (mouse bite) -> 0.80
        # Class 4 (spur) -> 0.75
        # Class 5 (spurious copper) -> 0.70
        if class_weights is None:
            self.class_weights = {
                0: 0.95,  # missing_hole / pin_hole
                1: 0.80,  # mouse_bite
                2: 0.90,  # open / scratch
                3: 1.00,  # short / solder_joint_defect
                4: 0.75,  # spur
                5: 0.70,  # spurious_copper
            }
        else:
            self.class_weights = class_weights

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gaussian RBF kernel computation between representations x and y"""
        # x: (N, D), y: (M, D)
        diff = torch.cdist(x, y, p=2)
        return torch.exp(-(diff**2) / (2.0 * (self.kernel_sigma**2) + 1e-8))

    def mmd_loss_per_class(self, x_source: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Compute MMD distance for a single class partition"""
        K_ss = self.rbf_kernel(x_source, x_source)
        K_tt = self.rbf_kernel(x_target, x_target)
        K_st = self.rbf_kernel(x_source, x_target)

        mmd_sq = torch.mean(K_ss) - 2.0 * torch.mean(K_st) + torch.mean(K_tt)
        return torch.sqrt(torch.clamp(mmd_sq, min=1e-6))

    def forward(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        labels_source: torch.Tensor = None,
        labels_target: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x_source: source domain features (N_s, D)
        x_target: target domain features (N_t, D)
        labels_source, labels_target: class labels (for filtering per-class)
        """
        if x_source.size(0) == 0 or x_target.size(0) == 0:
            return torch.tensor(0.0, device=x_source.device, requires_grad=True)

        # Global marginal alignment if class labels are absent or unlabelled
        if labels_source is None or labels_target is None:
            return self.mmd_loss_per_class(x_source, x_target)

        # Conditional class-wise alignment
        total_loss = torch.tensor(0.0, device=x_source.device)
        active_classes = 0

        for class_id in range(self.num_classes):
            mask_s = (labels_source == class_id)
            mask_t = (labels_target == class_id)

            if mask_s.sum() > 0 and mask_t.sum() > 0:
                x_s_c = x_source[mask_s]
                x_t_c = x_target[mask_t]

                mmd_c = self.mmd_loss_per_class(x_s_c, x_t_c)
                weight = self.class_weights.get(class_id, 1.0)
                total_loss = total_loss + weight * mmd_c
                active_classes += 1

        # Fallback to marginal MMD if no batch overlap exists
        if active_classes == 0:
            return self.mmd_loss_per_class(x_source, x_target)

        return total_loss / max(1, active_classes)

