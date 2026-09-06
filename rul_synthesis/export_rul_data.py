"""
export_rul_data.py
Processes synthetic degradation scenarios into tabular CSV for database loading,
hierarchical JSON for dashboard rendering, and generates lifecycle plots.
"""

import argparse
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
try:
    from .degradation_model import EquipmentDegradationModel
except (ImportError, ValueError):
    try:
        from rul_synthesis.degradation_model import EquipmentDegradationModel
    except (ImportError, ValueError):
        from degradation_model import EquipmentDegradationModel


def export_rul_datasets(
    scenarios_path: Path,
    output_dir: Path,
    reports_dir: Path,
):
    scenarios_path = Path(scenarios_path)
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not scenarios_path.exists():
        print(f"Scenarios file {scenarios_path} not found. Generating fresh batch...")
        model = EquipmentDegradationModel()
        scenarios = model.simulate_batch(num_scenarios=100, total_cycles=500)
        with open(scenarios_path, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2)
    else:
        with open(scenarios_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

    # 1. Export CSV for Database
    csv_file = output_dir / "rul_data.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["equipment_id", "cycle", "health_score", "defect_count", "rul_estimate"])

        for eqp in scenarios:
            eqp_id = eqp["equipment_id"]
            health_curve = eqp["health_curve"]
            defect_curve = eqp["defect_progression"]
            total_len = len(health_curve)

            for cycle_idx in range(total_len):
                h = round(health_curve[cycle_idx], 4)
                d = defect_curve[cycle_idx]
                rul = max(0, total_len - cycle_idx)
                writer.writerow([eqp_id, cycle_idx, h, d, rul])

    print(f"[OK] Exported database CSV: {csv_file}")

    # 2. Export Dashboard JSON (sample of equipment timelines)
    curves_file = output_dir / "rul_curves.json"
    dashboard_curves = []
    for eqp in scenarios[:15]:  # Select 15 representative machines
        timeline = []
        for c in range(0, len(eqp["health_curve"]), 5):  # Sample every 5 cycles for compact payload
            timeline.append({
                "cycle": c,
                "health": round(eqp["health_curve"][c], 3),
                "defects": eqp["defect_progression"][c],
                "rul": max(0, len(eqp["health_curve"]) - c),
            })
        dashboard_curves.append({
            "equipment_id": eqp["equipment_id"],
            "failure_cycle": eqp["failure_cycle"],
            "timeline": timeline,
        })

    with open(curves_file, "w", encoding="utf-8") as f:
        json.dump(dashboard_curves, f, indent=2)
    print(f"[OK] Exported dashboard curves JSON: {curves_file}")

    # 3. Visualization: Sample Equipment Degradation Curves
    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, eqp in enumerate(scenarios[:5]):
        cycles = range(len(eqp["health_curve"]))
        plt.plot(
            cycles,
            eqp["health_curve"],
            label=f"{eqp['equipment_id']} (λ={eqp['degradation_rate']:.4f})",
            color=colors[idx % len(colors)],
            linewidth=2,
        )

    plt.axhline(y=0.20, color="red", linestyle="--", label="Failure Threshold (h=0.20)")
    plt.title("Equipment Health Degradation Curves & Failure Threshold", fontsize=14, fontweight="bold")
    plt.xlabel("Operating Cycles (Production Runs)", fontsize=12)
    plt.ylabel("Health Score (h)", fontsize=12)
    plt.ylim(0.15, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    plot_path = reports_dir / "rul_sample_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[OK] Saved lifecycle plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Export RUL data for Database and Dashboard")
    parser.add_argument("--scenarios", type=str, default="data/rul_synthetic/scenarios.json")
    parser.add_argument("--output-dir", type=str, default="data/rul_synthetic")
    parser.add_argument("--reports-dir", type=str, default="reports")
    args = parser.parse_args()

    export_rul_datasets(Path(args.scenarios), Path(args.output_dir), Path(args.reports_dir))


if __name__ == "__main__":
    main()
