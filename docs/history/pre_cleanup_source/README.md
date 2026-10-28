# PCB Smart Maintenance — baseline

The Real-Minor dashboard, verified models and inspection database are connected on one local server. The browser runs real image inspections and displays saved results. Model accuracy work is deferred; startup never trains a model.

## Start the project

From this folder:

```sh
.venv/bin/python scripts/run_project.py
```

Open **http://127.0.0.1:8000**. The command verifies both model hashes, builds the existing dashboard, and starts the API serving the dashboard on the same port. SQLite persists inspections in `data/maintenance.db`. Stop with Ctrl+C. If the dashboard is already built, use `--skip-build` for faster startup.

The existing `.venv` is ready on this machine. For a fresh Python 3.11 environment, install `backend/requirements.txt`. Node.js/npm is needed to build the dashboard; the launcher uses the committed npm lockfile when dependencies are missing. A prebuilt `data/runtime/dashboard/` can run without Node via `--skip-build`.

## Use the connected dashboard

The dashboard uses the `Real-Minor` frontend, commit `50c12b383fe2f3453730772c02dd5884e09b3e6e`. The original source remains in `vendor/real_minor_frontend/`, verified against its SHA-256 manifest. API wiring lives separately in `dashboard_integration/` and is applied to a runtime copy during startup. The supplied theme, navigation, overview cards and charts are reused.

1. Open **Inspect a PCB** or **Inspections**.
2. Choose a JPEG/PNG and click **Run inspection**. Full bare board / Source is the default and supports all six labels. The DsPCBSD+ close-up / Target checkpoint supports only five labels and excludes missing_hole.
3. View the real annotated image, confidence values and pixel bounding boxes. Use **Original** to see the input or download the result JSON.
4. Open **Overview**. Counts, category charts, trends and recent inspections come from the database. Reloading keeps saved inspections and their images.
5. **Defect Analytics** shows saved class counts. **Alerts** lists recent short/open-circuit findings for review. **Settings** shows actual model evaluation records and a separate synthetic Phase 3 demonstration.

The equipment ID selects a separate inspection history. Confidence and model selection affect new uploads only. The dashboard refreshes every 30 seconds, after an upload, and when **Sync** is clicked. Totals cover 30 UTC calendar days; the recent-history and alert lists show at most 100 inspections. No detections does not certify a defect-free board.

Sample image: `examples/pcb.jpg`, also available at http://127.0.0.1:8000/demo/pcb.jpg. API documentation: http://127.0.0.1:8000/docs.

SQLite saves records in `data/maintenance.db`; uploaded normalized images and overlays are in `data/inspection_images/`. Keep both when backing up the project. Older and edge-only inspection records may have no image. Optional **Boards represented** is for actual known production exposure; leave it blank if unknown. Health and RUL remain unknown until enough known exposure is available.

## What is real and what is simulated

Image detections and database records come from the actual verified checkpoints. The target model's development mAP50 is 78.82%; it still misses defects and has imperfect localization. This is supervised target fine-tuning, not a successful MMD result.

The separate Phase 3 lifecycle data is synthetic. Live health/RUL stays unknown without declared inspection exposure and enough equipment history. Optional `boards_inspected` must describe actual represented exposure; do not invent it just to produce an estimate. With five contiguous known-exposure inspections covering at least 500 declared boards, the backend returns a simulator-based heuristic labelled `synthetic: true`, in cycles of 100 boards. It is not measured or calibrated physical machine lifetime.

## Components and paths

- Source detector: `phase1_final/model.pt`.
- Current target detector: `phase2/current_target/model.pt`.
- Simulator and dataset: `rul_synthesis/`, `data/phase3_rul/`.
- API and model registry: `backend/`.
- SQLite/PostgreSQL persistence: `database/`.
- Supplied Real-Minor React dashboard: `vendor/real_minor_frontend/`.
- Live frontend wiring: `dashboard_integration/`.
- Frontend source provenance: `vendor/real_minor_frontend.manifest.json`.
- Earlier local frontend retained unchanged: `frontend/`.
- Interactive API documentation: http://127.0.0.1:8000/docs.
- Canonical API routes use `/api/*`; the existing frontend and edge/CLI root routes remain available.

Local startup uses the launcher and port 8000. The Real-Minor frontend is compiled with its original Vite configuration and its own locked dependencies. The live integration is applied from `dashboard_integration/`; the original supplied files remain unchanged.

## PostgreSQL and Docker

The local zero-setup run uses SQLite. PostgreSQL integration was tested against an isolated PostgreSQL 17.8 cluster. Set an exported `DATABASE_URL` to use PostgreSQL, or run `.venv/bin/python scripts/run_project.py --check` followed by `docker compose up --build` for the supplied CPU API/dashboard and PostgreSQL services. Compose configuration is checked; Docker image builds are not part of local verification. `.env` is not loaded implicitly: export variables or use Uvicorn's `--env-file` option.

## Checks and previous work

```sh
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/run_project.py --check
.venv/bin/python phase4/verify_backend.py
```

`phase1_fast/`, `phase2/`, `phase3/` and `phase4/` retain training notebooks, actual results and reproducibility records. Earlier adaptation scripts are historical; the web API does not start training. Raspberry Pi/ONNX hardware deployment remains a separate task and is not claimed as tested by this local baseline.
