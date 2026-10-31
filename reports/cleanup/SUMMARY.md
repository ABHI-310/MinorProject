# Repository cleanup — 2026-10-26

Scope: `/Users/abhinavgupta/Developer/MinorProject` only. Downloads was not changed.
No model checkpoint, dataset, inspection record or historical experiment was
permanently deleted. The existing Python environment and Git internals were retained.

## Result

- Maintained commands are in `scripts/`, with one launcher: `scripts/manage.py`.
- Reusable Python packages are in `src/`; current frontend integration is in
  `frontend/`, with the original supplied source retained in `vendor/`.
- The current resumable Kaggle notebook is in `notebooks/`.
- Dataset, dependency and deployment configuration is in `configs/`.
- Original, prepared, synthetic, application and generated runtime data are
  separated under `data/`.
- Baseline models, pretrained weights, Kaggle packages and bulky verification
  outputs are separated under `artifacts/`.
- Superseded experiments, old datasets/packages and the alternate frontend are
  retained in `archive/legacy_repository/`. Small pre-cleanup source snapshots
  are retained in `archive/pre_cleanup_source/`.
- Root README, training/status docs, launcher paths, Docker paths, ignore rules
  and historical audit-path resolution were updated.

About 10 GB of historical material is archived inside the repository. Total disk
usage is largely unchanged: this cleanup organizes and isolates history without
irreversibly discarding experimental evidence.

## Verification

- Before cleanup: 85 tests passed. After cleanup: 88 tests passed (three new layout
  checks). Two pre-existing dependency deprecation warnings remain.
- Dashboard production build and model prerequisite check passed.
- Real source/target inference, temporary SQLite persistence and API contract
  checks passed; the existing inspection database was not used for these checks.
- Fresh full dataset audit: zero errors, the same five documented warnings.
- 11,413 historical artifact/input/image/label checksums passed using the relocation
  map. The version-2 Kaggle upload ZIP is byte-for-byte unchanged.
- The current resumable notebook is byte-for-byte unchanged from its source snapshot.
- Move verification accounted for 111,866 unchanged file records and 17 source-edit
  records with original snapshots. Seven compiled Python caches were regenerated
  by the checks. No unaccounted moved source/data files remain.
- One internal symlink was repaired after its target moved.
- Docker Compose configuration validation passed. No Docker image was built and
  no external service was deployed.

## Evidence and recovery

- `moves.json`: move inventory and original per-file hashes.
- `migration_verification.json`: unchanged/edited/generated-file accounting.
- `integrity.json`: historical seal and current package verification.
- `backend_check.json`: real-model API verification.
- `post_move_audit/`: fresh dataset audit.
- `symlink_updates.json`: the single repaired link.
- `configs/legacy_paths.json`: historical path mapping used by maintained tooling.

Current starting point: `README.md`; detailed directory policy:
`docs/REPOSITORY_LAYOUT.md`. Existing uploaded Kaggle data does not need replacing.
