# Phase 3 — synthetic RUL data

Run locally from the project root:

```sh
.venv/bin/python rul_synthesis/export_rul_data.py --scenarios 100 --cycles 500 --seed 42
```

No GPU, Kaggle, model download, or model training is required. The pipeline uses NumPy and Matplotlib. Generated data is in `data/phase3_rul/`, and the plot and validation report are in `reports/phase3/`. `config.RUL_DIR` points to this dataset. The old `data/rul_synthetic/` outputs are historical and were not overwritten.

## What is simulated

Each equipment lifecycle has an exponential health trend with noise, bounded between zero and its previous health. Failure occurs at the first simulated health at or below 0.2. Slow, normal, and fast profiles have different sampled decay rates. A seeded NumPy generator makes runs reproducible, and extending the observation horizon does not change earlier observations.

One production cycle assumes inspection of 100 boards. A health-dependent Poisson rate generates defect totals. Counts are allocated to the five shared detector classes: mouse bite, open circuit, short, spur, and spurious copper. Severity weights are illustrative assumptions, not measured engineering values. The target detector is not called during simulation, and no real equipment lifetime is inferred from PCB photographs.

## Correct RUL labels

When failure is observed, each row has `rul_cycles = failure_cycle - cycle`; the failure row has zero RUL. When the equipment has not failed by the observation horizon, `failure_cycle` and `rul_cycles` are unknown: JSON null and blank CSV fields. Such equipment is right-censored. Its lower bound is `observation_end_cycle + 1 - cycle`, since it remains unfailed through the last observation.

The earlier implementation incorrectly labelled the end of observation as a failure for unfailed equipment. This implementation preserves unknown outcomes instead of inventing a failure date.

## Files

- `scenarios.json`: simulated health, per-class defect counts, severity and failure metadata.
- `rul_data.csv`: all labelled cycle records, with explicit `synthetic` and censoring flags.
- `train.csv`, `val.csv`, `test.csv`: equipment-disjoint partitions, 70/15/15 equipment for the default 100-scenario run.
- `rul_curves.json`: dashboard-ready timelines and equipment metadata.
- `splits.json`: equipment-to-split assignments.
- `summary.json`: generation counts, assumptions and limitations.
- `manifest.json`: hashes of generated data files.
- `reports/phase3/rul_sample_curves.png`: health, defects and exact RUL/lower-bound examples.

The default seed produces 100 equipment lifecycles, 39,041 rows, 66 observed failures and 34 censored lifecycles. Seventy equipment belong to training, 15 to validation, and 15 to test. Never split individual rows randomly: that would leak the same equipment history across partitions.

## Using the dataset later

This phase generates data; it does not train or validate a real RUL predictor. Simulated health is a target/diagnostic quantity, not an available sensor input. Do not use health, terminal failure flags, failure cycle, observation end, exact RUL, lower bounds, profile or equipment ID as prediction features. Construct history windows only from defect observations at or before the current cycle, and keep equipment splits intact. Fit any scaling or thresholds using training equipment only. Exclude unknown exact labels from ordinary regression losses; never replace them with zero.

`estimate_rul_from_recent_defects` remains a transparent simulator-based demo heuristic for the existing backend interface. It assumes a known decay rate and exposure; its cycle estimate is not a learned or calibrated physical lifetime. It uses counts per inspection cycle. The existing backend's list of one entry per detected object must be replaced with properly aggregated equipment history in Phase 4. Empty history is an assumed healthy prior, not a measured healthy condition. The API/dashboard must label simulated estimates and insufficient history clearly.

## Checks

```sh
.venv/bin/python -m pytest tests/test_phase3_rul.py tests/test_mmd_and_rul.py -q
```

Tests cover first threshold crossing, exact and censored labels, reproducibility, horizon-invariant history, equipment-disjoint exports, invalid parameters and the demo heuristic. Dataset-level validation separately checks every generated row, split, count, label and hash. Neither these checks nor synthetic curves establish real-world RUL accuracy.
