# YOLO-PCB replacement validation — 2026-10-29

Decision: **do not replace the app's existing models with this checkpoint yet.**

YOLO-PCB is a promising PKU-family candidate. At 960 pixels and the author's example confidence of 0.90 it matched 531/562 source-validation defects, with one unmatched prediction and 31 misses (99.81% precision, 94.48% recall). At the UI's current 0.25 default it matched 561, with 190 unmatched predictions. Confidence scores are not calibrated between models. Source validation has shared PKU/TDD-Net ancestry with the external checkpoint; external training membership is unknown. These are development counts, not independent accuracy or mAP.

On all 403 DsPCBSD+ validation images (746 boxes, five supported classes), YOLO-PCB matched **zero defects at either 608 or 960 input size** at 0.25 confidence. At 608 there were three unmatched predictions; at 960 no predictions. Precision is undefined when there are no predictions; recall is zero. The app's source model also matched zero, while its target-trained model matched 591 with 366 unmatched predictions and 155 misses under controlled square preprocessing. This demonstrates failure to transfer across an in-scope imaging domain. Missing hole is untested in this domain. Exact byte overlap with source package images is zero, which does not prove independence from every external training source.

The candidate passed isolated API/direct prediction equality checks on one source image per class, finite in-bounds box checks, annotation encoding, database persistence through the existing API, and checkpoint identity. This uses a test registry injected into the app; active model loaders and checkpoints were not replaced. Legacy YOLOv5 classes require a compatible loader and module isolation; this is not a file-only swap.

Local CPU inference measurements and exploratory brightness/blur checks are recorded in replacement_report.json. The perturbation check used 12 familiar source images and 54 boxes, so it is not a new domain benchmark. Synthetic black/gray/white blanks produced no predictions at 0.90, but they do not establish false-positive performance on clean PCBs.

No independent six-class holdout or real clean-board false-positive benchmark is available. Training overlap, independent accuracy and clean-board behavior remain open. Do not use the strong source score in a paper as evidence of new-board generalization.

Next: retain current models, obtain a documented independent six-class set plus clean boards, and either restrict a future YOLO-PCB option to a demonstrated imaging domain or fine-tune it on the intended domain using group-disjoint splits. Select thresholds on validation only and keep the final test sealed. Do not infer machine remaining life from these results.

Reproduce from the repository root:

```sh
.venv/bin/python scripts/revamp/validate_yolo_pcb.py --dataset source
.venv/bin/python scripts/revamp/validate_yolo_pcb.py --dataset target
.venv/bin/python scripts/revamp/check_yolo_pcb_replacement.py
.venv/bin/python scripts/revamp/report_yolo_pcb_replacement.py
```

The validation scripts reuse cached predictions. Evidence: artifacts/verification/yolo_pcb_comparison, artifacts/verification/yolo_pcb_target and artifacts/verification/yolo_pcb_replacement. Model/source downloads and isolated seaborn dependency are in artifacts/external_models/yolo_pcb_review. Nothing in the live model registry was changed.
