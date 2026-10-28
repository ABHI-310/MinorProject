# Live API integration for Real-Minor

The supplied `vendor/real_minor_frontend/` is kept byte-for-byte as downloaded from GitHub. This folder contains the application wiring layered onto a generated copy during `scripts/run_project.py` startup. The original theme, navigation, overview cards, charts and recent-inspection layout are reused.

The integration replaces demo telemetry and simulated button handlers with `/api/dashboard`, `/api/detect` and saved image endpoints. Inspection includes a real uploader, original/annotated views, boxes, JSON download and honest maintenance status. Analytics and critical-defect review use saved records. Settings shows verified checkpoint metadata and an explicitly separate synthetic lifecycle demonstration.

Dashboard totals cover the selected equipment's last 30 UTC calendar days. History/review lists show at most the latest 100 inspections. Charts can show 7, 14 or 30 days. Each saved inspection is counted once even if it declares exposure covering multiple boards. Critical detection counts mean short/open-circuit findings, not acknowledged incidents or machine failures.

Source supports six classes. Target supports the five shared classes; missing-hole detections are excluded for target. Current selector and threshold affect new uploads only. Saved cards always retain the actual model and threshold used.

No model training runs at startup. No extra frontend service is required. The browser and API share port 8000. API outages show an error instead of substituting demo values. Older inspections created before image persistence may have no image preview.
