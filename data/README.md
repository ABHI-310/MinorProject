# Data

- `raw/pku/`: original PKU dataset bundle.
- `raw/dspcbsd/`: original DsPCBSD+ download/source bundle, including retained source metadata.
- `prepared/source/`: audited 480/120/93 source train/validation/development-test split.
- `prepared/target/`: audited 3606/403/992 target split; no missing-hole class.
- `prepared/revamp_v1/`: versioned full images/crops and separately reviewed background candidates.
- `synthetic/phase3_rul/`: synthetic lifecycle data, not measured machine wear.
- `application/`: inspection database and images; preserve together.
- `runtime/`: generated frontend, caches and build dependencies.

Dataset contents and labels were not changed by the directory cleanup. Existing
YAML may record historical absolute paths; use the maintained packaging commands
for fresh training configuration. See `configs/datasets.json` for provenance and
`configs/legacy_paths.json` to resolve paths in historical evidence.
