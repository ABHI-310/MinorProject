"""
degradation_model.py
Synthetic equipment degradation model linking PCB defect occurrence rates
to remaining useful life (RUL) estimation across production cycles.
Health decay model: h(t) = h_0 * exp(-λ*t) + ε
"""

import argparse
import json
from pathlib import Path
import numpy as np


class EquipmentDegradationModel:
    """
    Synthetic equipment degradation simulator:
    Simulates equipment calibration drift and mechanical wear over operating cycles.
    """

    def __init__(
        self,
        initial_health: float = 1.0,
        failure_threshold: float = 0.20,
        degradation_rate: float = 0.001,
        noise_level: float = 0.0001,
    ):
        self.initial_health = initial_health
        self.failure_threshold = failure_threshold
        self.degradation_rate = degradation_rate
        self.noise_level = noise_level

    def simulate_equipment_lifecycle(self, total_cycles: int = 500):
        """
        Simulates a single equipment lifecycle trajectory.
        Returns:
        - health: list of health scores [1.0 -> failure_threshold]
        - defects: list of defect counts per cycle
        - rul: remaining useful life in cycles
        - failure_cycle: cycle index at failure
        """
        health = [self.initial_health]
        defects = [0]

        for cycle in range(1, total_cycles):
            # Exponential decay with operational noise
            decay = np.exp(-self.degradation_rate * cycle)
            noise = np.random.normal(0, self.noise_level)
            new_health = self.initial_health * decay + noise
            new_health = float(np.clip(new_health, self.failure_threshold, 1.0))

            health.append(new_health)

            # Defect count increases inversely with equipment health
            defect_rate = 1.0 - new_health
            new_defects = int(50 * defect_rate) + int(np.random.poisson(2))
            defects.append(new_defects)

            # Check failure condition
            if new_health <= self.failure_threshold:
                rul = total_cycles - cycle
                return health, defects, rul, cycle

        return health, defects, 0, total_cycles

    def simulate_batch(self, num_scenarios: int = 100, total_cycles: int = 500, seed: int = 42):
        """Generates multiple realistic equipment degradation scenarios with parameter variations"""
        np.random.seed(seed)
        scenarios = []

        for i in range(num_scenarios):
            # Degradation rate varies by machine load & operating conditions
            deg_rate = float(np.clip(np.random.normal(0.001, 0.0002), 0.0005, 0.003))
            noise = float(np.random.uniform(0.00005, 0.0003))

            model = EquipmentDegradationModel(
                initial_health=1.0,
                failure_threshold=0.20,
                degradation_rate=deg_rate,
                noise_level=noise,
            )

            health, defects, rul, fail_cycle = model.simulate_equipment_lifecycle(total_cycles)

            scenarios.append({
                "equipment_id": f"EQP_{i:03d}",
                "health_curve": health,
                "defect_progression": defects,
                "rul_cycles": rul,
                "failure_cycle": fail_cycle,
                "degradation_rate": deg_rate,
            })

        return scenarios


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic equipment RUL scenarios")
    parser.add_argument("--output-dir", type=str, default="data/rul_synthetic")
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = EquipmentDegradationModel()
    scenarios = model.simulate_batch(num_scenarios=args.num_scenarios, total_cycles=args.cycles)

    out_path = out_dir / "scenarios.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)

    print(f"✓ Successfully generated {len(scenarios)} equipment scenarios in {out_path}")


if __name__ == "__main__":
    main()

