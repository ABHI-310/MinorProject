# PCB inspection and model revamp

University project: detect six bare-PCB defect classes, retain inspection history,
and demonstrate explicitly synthetic degradation trends. Camera/edge deployment
and real-machine lifetime estimation are future scope.

The existing app works with preserved baseline models. The revamp has completed
its dataset audit and Kaggle learning/resume preflight. Full candidate training is
**Phase 3**, not yet completed; the six-image preflight score is not model accuracy.

## Start the app

From this repository:

```sh
.venv/bin/python scripts/manage.py app
```

Open **http://127.0.0.1:8000**. Startup verifies model hashes, builds the dashboard,
and serves the API and frontend. Use `app --skip-build` to reuse the existing build
or `app --check` to verify/build without starting a server. Startup does not train.

The local `.venv` is preserved. For a fresh Python 3.11 environment, install
`requirements.txt`; it includes `configs/requirements.txt`. Node/npm is needed to
build the dashboard. SQLite is the default; environment variables must be exported
or passed explicitly—`.env` is not loaded automatically.

## Where everything lives

```text
scripts/          Maintained commands; scripts/revamp/ is the training workflow
src/              Backend, database, model utilities, preprocessing and simulator
frontend/         Live dashboard integration applied to the supplied frontend
vendor/           Unchanged supplied frontend and its provenance manifest
notebooks/        Current notebook to import into Kaggle
configs/          Dataset catalog, dependencies, Dockerfile and historical path map
data/
  raw/            Original dataset bundles
  prepared/       Audited source, target and versioned training views
  synthetic/      Explicitly synthetic degradation data
  application/    Inspection database and saved inspection images
  runtime/        Generated dashboard, caches and build dependencies
artifacts/
  models/         Preserved source and target baseline packages
  pretrained/     COCO initialization weights
  kaggle/         Versioned upload ZIPs and their immutable expanded payloads
  verification/  Previous smoke-run checkpoints, logs and extracted test packages
reports/          Audits, results, provenance and cleanup verification
examples/         Demo PCB image
docs/             Current training plan, status and repository guide
  history/        Original specifications and small provenance source snapshots
tests/            Automated tests
```

The superseded archive was removed with user authorization. Small provenance
snapshots remain in `docs/history/pre_cleanup_source/`, and historical regression
helpers in `tests/legacy_support/`. Current datasets, baselines, checkpoints and
Kaggle files are retained. Downloads was not changed. See
[the layout guide](docs/REPOSITORY_LAYOUT.md) for details.

## Current Kaggle files

- Next step: [Full-board-only experiment 2](notebooks/PCB_Revamp_Phase3_Full_Boards_Kaggle.ipynb)
- Completed experiment 1: [Mixed-view source training](notebooks/PCB_Revamp_Phase3_Source_Training_Kaggle.ipynb)
- Verified GPU pilot: [Phase 3 source pilot](notebooks/PCB_Revamp_Phase3_Pilot_Kaggle.ipynb)
- Completed preflight: [Phase 2 smoke/resume notebook](notebooks/PCB_Revamp_Phase2_Resume_Kaggle.ipynb)
- Dataset: `artifacts/kaggle/phase2_v2/pcb_revamp_phase2.zip`
- Instructions: [training and checkpoint guide](docs/TRAINING.md)
- Progress: [implementation status](docs/STATUS.md)

The ZIP and notebook bytes were preserved during cleanup. An already uploaded
Kaggle dataset does **not** need replacing. The notebook saves every completed
training epoch; download its checkpoint before ending a session. The Phase 2 notebook is only a preflight; the Phase 3 pilot has now been verified on T4. The full source-training notebook
runs up to 60 epochs with early stopping; review its results before model promotion.

## Commands and checks

```sh
.venv/bin/python scripts/manage.py --help
.venv/bin/python scripts/manage.py test
.venv/bin/python scripts/manage.py verify-layout
.venv/bin/python scripts/manage.py verify-api --report reports/local_api_check.json
.venv/bin/python scripts/manage.py audit --output reports/new_audit_version
```

`prepare`, `package`, `notebook` and `smoke` are also available through `manage.py`.
Pass their own `--help` after the command. Use a new version directory for data and
packages. Tests can also run with `.venv/bin/python -m pytest tests -q`.

## App data and model limits

Back up **both** `data/application/maintenance.db` and
`data/application/inspection_images/`. The current models are
`artifacts/models/source/model.pt` and `artifacts/models/target/model.pt`. Source
supports six classes; target has no missing-hole training/evaluation support.

Detections and inspection histories are real model outputs. The degradation/RUL
component remains a synthetic demonstration, not validated machine lifetime.
Historical test sets are development comparisons. No detections does not certify
that a board is defect-free. Detailed constraints are in the training guide.

For PostgreSQL/Docker, build the dashboard first with `app --check`, then use
`docker compose up --build`. Docker build paths have been updated; an actual image
build is a separate environment check.
