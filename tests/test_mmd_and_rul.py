from __future__ import annotations

import torch

from models.mmd_loss import DefectAwareMMDLoss
from rul_synthesis.degradation_model import EquipmentDegradationModel, estimate_rul_from_recent_defects


def test_mmd_is_near_zero_for_identical_features():
    values = torch.randn(4, 8)
    assert DefectAwareMMDLoss()(values, values).item() < 1e-5


def test_rul_model_exports_nonnegative_lifecycle():
    lifecycle = EquipmentDegradationModel(seed=10).simulate_equipment_lifecycle(total_cycles=50)
    assert len(lifecycle.health_curve) == len(lifecycle.defect_progression)
    health, rul = estimate_rul_from_recent_defects(lifecycle.defect_progression)
    assert 0 <= health <= 1 and rul >= 0
