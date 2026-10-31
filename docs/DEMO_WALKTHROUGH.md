# Semester project demonstration

1. Start: `.venv/bin/python scripts/manage.py app`. Open http://127.0.0.1:8000 (or port 8010 for the current running session).
2. Use a distinct equipment ID such as PRESENTATION_01. Select COCO → K500 and confidence 0.25.
3. Upload images from `examples/k500_demo/` in numerical order. Each image has a matching YOLO label file; expected defect counts and original provenance are in `manifest.json`. One image can contain several classes. Enter 1 board represented for a single-board image; leave unknown counts blank. Do not invent exposure to obtain an RUL value.
4. Compare predicted labels and boxes against the saved annotations. Include misses and false detections. These are familiar validation images for a repeatable UI check; do not claim independent accuracy from them.
5. Show Inspections and Defect Analytics. Counts are model findings and include false detections; absence of predictions is not a certified clean-board result.
6. Open Settings and review Models and Maintenance. Different model cards use different evaluation sets and label budgets; compare using the saved controlled reports, not arbitrary card rankings.
7. Under Inspection history · maintenance review, click Run synthetic review demo. Fabricated counts 1,1,1,4,4,4 across equal 100-board observations trigger REVIEW REQUIRED. This demo writes no inspection history.
8. Load the synthetic lifecycle below it to explain the simulated health/RUL assumptions. Do not present these as real machine-life estimates. The live-history outlook requires at least five exposure-known inspections and 500 reported boards; normal single-board uploads will not meet that immediately.

## Results to explain

The best current 500-label candidate improves over PCB-source initialization, but 100-label performance remains substantially weaker. External green soldered-board and binary DeepPCB examples exposed generalization failures. The proposed weighted adapter has not demonstrated a benefit. Hardware and real failure-data validation are future work.

## Evidence

- `reports/revamp/k500_coco_validation/report.json`
- `reports/revamp/k500_coco_development/report.json`
- `reports/revamp/k500_evaluation_data_audit.json`
- `reports/revamp/k100_coco_download_verification.json`
- `examples/k500_demo/manifest.json`
