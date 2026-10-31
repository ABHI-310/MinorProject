"""Reproducible synthetic lifecycles; no real equipment failure calibration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

# Illustrative simulator assumptions, not measured physical severities.
DEFECT_SEVERITY = {"mouse_bite": 1.0, "open_circuit": 1.5, "short": 1.5,
                   "spur": 0.75, "spurious_copper": 0.75}
BACKGROUND_DEFECTS = 0.15
DEGRADATION_DEFECT_SCALE = 12.0


@dataclass
class Lifecycle:
    equipment_id: str
    health_curve: list[float]
    defect_progression: list[int]
    rul_cycles: int | None
    failure_cycle: int | None
    degradation_rate: float
    failure_observed: bool
    observation_end_cycle: int
    failure_threshold: float
    defect_counts_by_type: list[dict[str, int]]
    severity_progression: list[float]
    profile: str
    boards_per_cycle: int = 100
    synthetic: bool = True


def _positive_integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


class EquipmentDegradationModel:
    """Exponential health trend with bounded noise and no spontaneous recovery.

    total_cycles counts observations including cycle zero. Failure is the FIRST
    observed health <= threshold. A lifecycle ending earlier is right-censored:
    failure_cycle and exact terminal RUL are unknown, represented by None.
    One production cycle represents inspection of 100 boards by assumption.
    """

    def __init__(self, initial_health: float = 1.0, failure_threshold: float = 0.2,
                 degradation_rate: float = 0.004, noise_level: float = 0.003,
                 seed: int | None = None):
        values = [initial_health, failure_threshold, degradation_rate, noise_level]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Simulation parameters must be finite")
        if not 0 < failure_threshold < initial_health <= 1:
            raise ValueError("Require 0 < failure_threshold < initial_health <= 1")
        if degradation_rate <= 0 or not 0 <= noise_level <= 0.1:
            raise ValueError("degradation_rate must be positive; noise_level must be in [0, 0.1]")
        self.initial_health = initial_health
        self.failure_threshold = failure_threshold
        self.degradation_rate = degradation_rate
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    def simulate_equipment_lifecycle(self, equipment_id: str = "EQP_000",
                                     total_cycles: int = 500, profile: str = "custom") -> Lifecycle:
        _positive_integer(total_cycles, "total_cycles", 2)
        if not equipment_id:
            raise ValueError("equipment_id must not be empty")
        health: list[float] = []
        defects: list[int] = []
        counts_by_type: list[dict[str, int]] = []
        severity: list[float] = []
        failure_cycle = None
        for cycle in range(total_cycles):
            if cycle == 0:
                current = self.initial_health
            else:
                ideal = self.initial_health * math.exp(-self.degradation_rate * cycle)
                current = float(np.clip(ideal + self.rng.normal(0, self.noise_level), 0, health[-1]))
            health.append(current)
            # Independent observations around a health-dependent rate; not cumulative detections.
            expected = BACKGROUND_DEFECTS + DEGRADATION_DEFECT_SCALE * (1 - current) ** 2
            total = int(self.rng.poisson(expected))
            per_type = self.rng.multinomial(total, [0.2] * 5)
            counts = {name: int(n) for name, n in zip(DEFECT_SEVERITY, per_type)}
            defects.append(total)
            counts_by_type.append(counts)
            severity.append(sum(DEFECT_SEVERITY[name] * n for name, n in counts.items()))
            if current <= self.failure_threshold:
                failure_cycle = cycle
                break
        return Lifecycle(
            equipment_id=equipment_id, health_curve=health, defect_progression=defects,
            rul_cycles=0 if failure_cycle is not None else None, failure_cycle=failure_cycle,
            degradation_rate=self.degradation_rate, failure_observed=failure_cycle is not None,
            observation_end_cycle=len(health) - 1, failure_threshold=self.failure_threshold,
            defect_counts_by_type=counts_by_type, severity_progression=severity, profile=profile,
        )

    def simulate_batch(self, num_scenarios: int = 100, total_cycles: int = 500) -> list[Lifecycle]:
        _positive_integer(num_scenarios, "num_scenarios")
        _positive_integer(total_cycles, "total_cycles", 2)
        scenarios = []
        profiles = [("slow", 0.625), ("normal", 1.0), ("fast", 1.5)]
        for index in range(num_scenarios):
            profile, multiplier = profiles[index % len(profiles)]
            mean_rate = self.degradation_rate * multiplier
            rate = float(max(0.0001, self.rng.normal(mean_rate, mean_rate * 0.1)))
            noise = self.noise_level * float(self.rng.uniform(0.5, 1.5))
            simulator = EquipmentDegradationModel(
                self.initial_health, self.failure_threshold, rate, min(noise, 0.1),
                int(self.rng.integers(2**31)),
            )
            scenarios.append(simulator.simulate_equipment_lifecycle(f"EQP_{index:03d}", total_cycles, profile))
        return scenarios

    @staticmethod
    def to_dict(scenario: Lifecycle) -> dict:
        return asdict(scenario)


def estimate_rul_from_recent_defects(defect_counts: list[int], baseline_rate: float = 1.0,
                                     max_cycles: int = 500) -> tuple[float, int]:
    """Legacy two-value demo interface; ONLY a simulator-based heuristic.

    Each input must be a total defect count for one equal-exposure inspection
    cycle, not one element per detected object. Assumes 100 boards per cycle,
    known exponential decay 0.004 and threshold 0.2. No fitted RUL model and no
    real failure prediction is implied. Empty history returns the assumed
    healthy prior. Phase 4 must label this estimate as simulated and distinguish
    insufficient history from a real measurement.
    """
    _positive_integer(max_cycles, "max_cycles")
    if not math.isfinite(baseline_rate) or baseline_rate <= 0:
        raise ValueError("baseline_rate must be finite and positive")
    values = np.asarray(defect_counts, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Defect counts must be a one-dimensional sequence of finite nonnegative values")
    mean = float(values[-20:].mean()) / baseline_rate if len(values) else 0.0
    health = float(np.clip(1 - math.sqrt(max(0, mean - BACKGROUND_DEFECTS) / DEGRADATION_DEFECT_SCALE), 0, 1))
    remaining = math.log(health / 0.2) / 0.004 if health > 0.2 else 0.0
    return health, min(max_cycles, max(0, round(remaining)))
