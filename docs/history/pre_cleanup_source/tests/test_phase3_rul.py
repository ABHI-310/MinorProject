"""Check temporal labels and split boundaries, not simulator prediction accuracy."""
import csv
import json
import math
from pathlib import Path

import pytest

from rul_synthesis.degradation_model import EquipmentDegradationModel, estimate_rul_from_recent_defects
from rul_synthesis.export_rul_data import export, lifecycle_rows


def test_exact_rul_uses_first_failure_not_requested_horizon():
    model = EquipmentDegradationModel(degradation_rate=.5, noise_level=0, seed=1)
    item = model.simulate_equipment_lifecycle(total_cycles=50)
    assert item.failure_cycle == math.ceil(math.log(5)/.5)
    assert item.failure_observed and item.rul_cycles == 0
    rows = lifecycle_rows(model.to_dict(item), 'train')
    assert [r['rul_cycles'] for r in rows] == list(range(item.failure_cycle, -1, -1))
    assert all(h > .2 for h in item.health_curve[:-1]) and item.health_curve[-1] <= .2


def test_healthy_horizon_is_censored_not_failure():
    model = EquipmentDegradationModel(degradation_rate=.001, noise_level=0, seed=1)
    item = model.simulate_equipment_lifecycle(total_cycles=50)
    assert not item.failure_observed and item.failure_cycle is None and item.rul_cycles is None
    rows = lifecycle_rows(model.to_dict(item), 'test')
    assert all(row['rul_cycles'] is None for row in rows)
    assert rows[0]['rul_lower_bound_cycles'] == 50
    assert rows[-1]['rul_lower_bound_cycles'] == 1


def test_simulation_is_reproducible_and_horizon_does_not_change_past():
    short = EquipmentDegradationModel(seed=42).simulate_batch(6, 40)
    same = EquipmentDegradationModel(seed=42).simulate_batch(6, 40)
    long = EquipmentDegradationModel(seed=42).simulate_batch(6, 500)
    assert short == same
    for prefix, complete in zip(short, long):
        assert prefix.health_curve == complete.health_curve[:40]
        assert prefix.defect_progression == complete.defect_progression[:40]
        assert all(a >= b for a, b in zip(complete.health_curve, complete.health_curve[1:]))
        assert all(sum(c.values()) == n for c, n in zip(complete.defect_counts_by_type, complete.defect_progression))


def test_export_separates_equipment_and_preserves_unknown_labels(tmp_path):
    out = tmp_path/'data'
    summary = export(12, 50, 7, out, tmp_path/'reports')
    assert summary['censored'] == 12 and summary['failure_observed'] == 0
    assignments = json.loads((out/'splits.json').read_text())['assignments']
    groups = {}
    for split in ['train', 'val', 'test']:
        with (out/f'{split}.csv').open() as handle:rows = list(csv.DictReader(handle))
        groups[split] = {r['equipment_id'] for r in rows}
        assert groups[split]
        assert all(r['split'] == split and r['synthetic'] == 'True' for r in rows)
        assert all(r['rul_cycles'] == '' and r['failure_cycle'] == '' for r in rows)
    assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
    assert set.union(*groups.values()) == set(assignments)
    assert (tmp_path/'reports/rul_sample_curves.png').is_file()


@pytest.mark.parametrize('kwargs', [{'failure_threshold':1.}, {'degradation_rate':0}, {'noise_level':-.01}])
def test_invalid_simulation_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):EquipmentDegradationModel(**kwargs)


def test_demo_heuristic_falls_with_defect_burden_and_rejects_invalid_counts():
    healthy = estimate_rul_from_recent_defects([0]*20)
    unhealthy = estimate_rul_from_recent_defects([8]*20)
    assert healthy[0] > unhealthy[0] and healthy[1] > unhealthy[1]
    with pytest.raises(ValueError):estimate_rul_from_recent_defects([-1])
