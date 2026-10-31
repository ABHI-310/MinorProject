"""Generate checked synthetic RUL data; run locally, no GPU required."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rul_synthesis.degradation_model import EquipmentDegradationModel, DEFECT_SEVERITY
from config import RUL_DIR, REPORTS_DIR


def equipment_splits(ids: list[str], seed: int) -> dict[str, str]:
    if len(ids) < 3 or len(set(ids)) != len(ids):
        raise ValueError("Need at least three unique equipment IDs for train/validation/test splits")
    order = np.random.default_rng(seed).permutation(sorted(ids)).tolist()
    validation = max(1, int(len(order) * 0.15))
    test = max(1, int(len(order) * 0.15))
    train = len(order) - validation - test
    return {equipment: split for split, group in (
        ("train", order[:train]), ("val", order[train:train + validation]),
        ("test", order[train + validation:])) for equipment in group}


def lifecycle_rows(item: dict, split: str) -> list[dict]:
    length = len(item['health_curve'])
    if not all(len(item[key]) == length for key in
               ['defect_progression', 'severity_progression', 'defect_counts_by_type']):
        raise ValueError("Lifecycle arrays have different lengths")
    if item['failure_observed']:
        if item['failure_cycle'] != length - 1 or item['health_curve'][-1] > item['failure_threshold']:
            raise ValueError("Observed failure is inconsistent with the health curve")
        if any(h <= item['failure_threshold'] for h in item['health_curve'][:-1]):
            raise ValueError("Failure must occur at the first threshold crossing")
    elif item['failure_cycle'] is not None or any(h <= item['failure_threshold'] for h in item['health_curve']):
        raise ValueError("Censored equipment must not have an observed threshold crossing")
    rows = []
    for cycle, (health, defects) in enumerate(zip(item['health_curve'], item['defect_progression'])):
        rul = item['failure_cycle'] - cycle if item['failure_observed'] else None
        row = dict(equipment_id=item['equipment_id'], split=split, cycle=cycle,
                   health_score=health, defect_count=defects, severity_score=item['severity_progression'][cycle],
                   rul_cycles=rul, rul_lower_bound_cycles=rul if rul is not None else length - cycle,
                   failure_observed=item['failure_observed'], failure_cycle=item['failure_cycle'],
                   observation_end_cycle=length - 1, profile=item['profile'],
                   boards_per_cycle=item['boards_per_cycle'], synthetic=True)
        row.update({f'defects_{name}': n for name, n in item['defect_counts_by_type'][cycle].items()})
        rows.append(row)
    return rows


def export(num_scenarios: int = 100, cycles: int = 500, seed: int = 42,
           output: Path | None = None, reports: Path | None = None) -> dict:
    if num_scenarios < 3:
        raise ValueError("At least three scenarios are required")
    output = Path(output) if output is not None else RUL_DIR
    reports = Path(reports) if reports is not None else REPORTS_DIR/'phase3'
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    scenarios = EquipmentDegradationModel(seed=seed).simulate_batch(num_scenarios, cycles)
    serialised = [EquipmentDegradationModel.to_dict(item) for item in scenarios]
    splits = equipment_splits([item['equipment_id'] for item in serialised], seed)
    rows, curves = [], []
    for item in serialised:
        points = lifecycle_rows(item, splits[item['equipment_id']]); rows.extend(points)
        curves.append(dict(equipment_id=item['equipment_id'], synthetic=True,
                           split=splits[item['equipment_id']], failure_observed=item['failure_observed'],
                           failure_cycle=item['failure_cycle'], timeline=points))
    def write_json(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    write_json('scenarios.json', dict(synthetic=True, seed=seed, scenarios=serialised))
    write_json('rul_curves.json', dict(synthetic=True, rul_units='production cycles', curves=curves))
    write_json('splits.json', dict(seed=seed, split_unit='equipment_id', assignments=splits))
    for split in [None, 'train', 'val', 'test']:
        with (output/('rul_data.csv' if split is None else f'{split}.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader()
            writer.writerows(row for row in rows if split is None or row['split'] == split)
    summary = dict(synthetic=True, seed=seed, equipment=len(scenarios), rows=len(rows),
                   observation_horizon_cycles=cycles,
                   failure_observed=sum(item.failure_observed for item in scenarios),
                   censored=sum(not item.failure_observed for item in scenarios),
                   equipment_per_split=dict(Counter(splits.values())),
                   rows_per_split=dict(Counter(row['split'] for row in rows)),
                   output_directory=str(output.resolve()),
                   exact_rul_definition='first observed threshold-crossing cycle minus current cycle',
                   censoring='Unobserved failures: exact RUL null/blank; lower bound = observation_end_cycle + 1 - cycle.',
                   severity_weights=DEFECT_SEVERITY,
                   assumptions=['Health has an exponential trend with noise and no spontaneous recovery.',
                                'Each cycle inspects 100 boards; defect counts are Poisson observations.',
                                'Five defect types are allocated uniformly; severity weights are illustrative.',
                                'Different equipment IDs are placed in different data splits.'],
                   limitations=['No real equipment run-to-failure data was used.',
                                'PCB defect detections do not establish the cause or lifetime of production equipment.',
                                'No learned RUL predictor or measured real-world RUL accuracy is claimed.',
                                'Latent health, failure cycle, lower-bound labels, profile and equipment ID must not be used as learned prediction features.'])
    write_json('summary.json', summary)
    write_json('manifest.json', dict(sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'}))
    plot_curves(curves, reports/'rul_sample_curves.png', seed)
    (reports/'generation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def plot_curves(curves: list[dict], path: Path, seed: int = 42) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Include both observed failures and censored examples where available.
    selected = []
    for observed in [True, False]:
        selected.extend([c for c in curves if c['failure_observed'] == observed][:2])
    selected.extend([c for c in curves if c['equipment_id'] not in {s['equipment_id'] for s in selected}][:5-len(selected)])
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for curve in selected:
        points = curve['timeline']; x = [p['cycle'] for p in points]
        label = curve['equipment_id'] + (' (failure observed)' if curve['failure_observed'] else ' (censored)')
        axes[0].plot(x, [p['health_score'] for p in points], label=label)
        axes[1].plot(x, [p['defect_count'] for p in points], alpha=.65)
        axes[2].plot(x, [p['rul_cycles'] if p['rul_cycles'] is not None else p['rul_lower_bound_cycles'] for p in points],
                     linestyle='-' if curve['failure_observed'] else '--')
    axes[0].axhline(.2, color='black', linestyle=':', label='Simulated failure threshold')
    axes[0].set(ylabel='Simulated health (0–1)', title='Synthetic equipment health trajectories')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[1].set(ylabel='Defects / 100 boards', title='Simulated defect observations per production cycle')
    axes[2].set(xlabel='Production cycle', ylabel='Remaining cycles', title='Exact simulated RUL (solid) or lower bound (dashed)')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Phase 3 — simulated data only; no measured equipment lifetimes', fontsize=12)
    fig.text(.02, .01, f'Source: local simulator, seed {seed}. Each line is one equipment lifecycle; cycle ranges vary.', fontsize=8)
    fig.tight_layout(rect=(0, .025, 1, .97));fig.savefig(path, dpi=150);plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', type=int, default=100)
    parser.add_argument('--cycles', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=RUL_DIR)
    parser.add_argument('--reports', type=Path, default=REPORTS_DIR/'phase3')
    args = parser.parse_args()
    print(json.dumps(export(args.scenarios, args.cycles, args.seed, args.output, args.reports), indent=2))
