# Repository layout and cleanup policy

Cleanup scope: this repository only, 2026-10-26. Downloads, other projects, the
virtual environment and Git's internal database were not cleaned or modified.

## Canonical locations

- Runnable commands live under `scripts/`; `scripts/manage.py` sets import paths
  consistently for this checkout and child Python processes.
- Reusable Python modules live under `src/`. `src/config.py` defines the repository
  root and runtime paths. The six-class training taxonomy is in `configs/datasets.json`.
- `frontend/` contains live integration files. The original supplied React source
  stays in `vendor/real_minor_frontend/` and is checked against its SHA-256 manifest.
  Startup builds a runtime copy with the integration applied; it never edits vendor.
- `data/raw/` and `data/prepared/` are retained inputs. Do not overwrite a dataset
  version or move validation images into training.
- `data/application/` holds irreplaceable local inspection history; back up the
  database and images together. `data/runtime/` holds generated caches/builds.
- `artifacts/models/` contains current baseline packages. `artifacts/pretrained/`
  holds initialization weights. Training output belongs in a new version under
  `artifacts/training/`; Kaggle upload packages belong under `artifacts/kaggle/`.
- `notebooks/` contains the currently recommended notebook; historical notebooks
  are not part of the active workflow.
- Keep concise reviewable reports under `reports/`; bulky verification checkpoints
  and extracted packages live under `artifacts/verification/`.

## Historical work

The user authorized removal of the superseded archive on 2026-10-26. Old
experiments, alternate frontend, redundant dataset exports and generated logs
were removed. Current datasets, baseline models, application history, Kaggle
packages and verification checkpoints remain in their canonical locations.

`docs/history/pre_cleanup_source/` preserves the small original source/configuration
snapshot, including the exact files sealed by the Phase 1 audit.
`tests/legacy_support/accuracy/` preserves five historical Python helpers needed
by regression tests; these are not current training entrypoints.

## Provenance and recovery

- `reports/cleanup/moves.json` records old/new paths and the SHA-256 inventories
  captured before each move.
- `configs/legacy_paths.json` resolves old report paths to relocated data or source
  snapshots. Immutable old report JSON and the Kaggle package are not rewritten.
- `scripts/manage.py verify-layout` checks the original Phase 1 seals, all audited
  prepared images/labels, and the exact existing Kaggle package.
- `reports/cleanup/archive_removal.json` records retained dependencies and removed
  path mappings. The old move manifest is historical; deleted archive entries
  cannot be restored from it. Source snapshots are under
  `docs/history/pre_cleanup_source/`.

Archived logs/YAML can contain old absolute local paths. Those describe the original
run. For a new run use the maintained preparation/package commands, which generate
new writable configuration files from the current manifest.
